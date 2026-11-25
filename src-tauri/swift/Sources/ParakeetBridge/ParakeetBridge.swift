// C ABI over FluidAudio's batch Parakeet TDT transcription, for the Rust side (src/parakeet.rs).
//
// Every call blocks the calling thread until the async FluidAudio work finishes. While it waits,
// the calling thread polls `callback` every 50 ms with the latest progress, and the callback's
// return value cancels the work. All callbacks run on the calling thread, never on Swift's.
//
// Strings returned by this library are malloc'd; free them with `ptb_free_string`.

import CoreML
import FluidAudio
import Foundation

/// `callback(ctx, fraction, text)`: fraction in [0, 1] or -1 when unknown, text may be NULL.
/// Returns true to cancel.
public typealias PtbCallback = @convention(c) (UnsafeMutableRawPointer?, Double, UnsafePointer<CChar>?) -> Bool

final class Engine {
    let manager: AsrManager
    init(manager: AsrManager) { self.manager = manager }
}

/// Latest progress, written from any Swift thread and read by the polling thread.
final class ProgressBox: @unchecked Sendable {
    private let lock = NSLock()
    private var fraction: Double = -1
    private var text: String?
    private var version = 0
    private var windows: [Int: (start: Double, end: Double, text: String)] = [:]
    private var windowCount = 0

    func set(fraction: Double, text: String?) {
        lock.lock()
        defer { lock.unlock() }
        self.fraction = fraction
        self.text = text
        version += 1
    }

    /// Windows can finish out of order; report the text of the contiguous prefix that's done.
    func addWindow(_ w: AsrManager.WindowProgress) {
        lock.lock()
        defer { lock.unlock() }
        windows[w.index] = (w.startSeconds, w.endSeconds, w.text)
        windowCount = w.windowCount
        fraction = Double(windows.count) / Double(max(1, windowCount))
        // Window texts overlap at the seams; the Rust side merges them (parakeet::merge_windows).
        text = prefix().map { $0.text }.joined(separator: "\u{1E}")
        version += 1
    }

    private func prefix() -> [(start: Double, end: Double, text: String)] {
        var out: [(start: Double, end: Double, text: String)] = []
        var i = 0
        while let w = windows[i] {
            out.append(w)
            i += 1
        }
        return out
    }

    func completedWindows() -> [(start: Double, end: Double, text: String)] {
        lock.lock()
        defer { lock.unlock() }
        return prefix()
    }

    func snapshot(since seen: inout Int) -> (Double, String?)? {
        lock.lock()
        defer { lock.unlock() }
        guard version != seen else { return nil }
        seen = version
        return (fraction, text)
    }
}

final class ResultBox<T>: @unchecked Sendable {
    var value: Result<T, Error>?
}

/// Run `op` to completion while polling `callback` on this thread.
func runBlocking<T>(
    progress: ProgressBox, callback: PtbCallback?, ctx: UnsafeMutableRawPointer?,
    _ op: @escaping @Sendable () async throws -> T
) -> (result: Result<T, Error>, cancelled: Bool) {
    let done = DispatchSemaphore(value: 0)
    let box = ResultBox<T>()
    // Explicit priority matters: a Task started from a plain (Rust) thread otherwise runs at a
    // low QoS, and Core ML then schedules the Neural Engine work about 2x slower.
    let task = Task.detached(priority: .userInitiated) {
        do { box.value = .success(try await op()) } catch { box.value = .failure(error) }
        done.signal()
    }
    var seen = 0
    var cancelled = false
    while done.wait(timeout: .now() + .milliseconds(50)) == .timedOut {
        guard let callback, !cancelled else { continue }
        let snap = progress.snapshot(since: &seen)
        let stop: Bool
        if let (fraction, text) = snap, let text {
            stop = text.withCString { callback(ctx, fraction, $0) }
        } else {
            stop = callback(ctx, snap?.0 ?? -1, nil)
        }
        if stop {
            cancelled = true
            task.cancel()
        }
    }
    return (box.value!, cancelled)
}

func setError(_ out: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?, _ error: Error) {
    let message = (error as? LocalizedError)?.errorDescription ?? String(describing: error)
    out?.pointee = strdup(message)
}

func jsonString(_ object: Any) -> UnsafeMutablePointer<CChar>? {
    guard let data = try? JSONSerialization.data(withJSONObject: object),
        let s = String(data: data, encoding: .utf8)
    else { return nil }
    return strdup(s)
}

/// Load (downloading on first use) Parakeet TDT `version` (2 = English, 3 = multilingual).
/// Models already in FluidAudio's own cache are reused; otherwise they're downloaded to
/// `models_dir`. Progress text is "download" or "compile". Returns NULL and sets `error` on failure.
@_cdecl("ptb_load")
public func ptb_load(
    _ version: Int32,
    _ modelsDir: UnsafePointer<CChar>?,
    _ callback: PtbCallback?,
    _ ctx: UnsafeMutableRawPointer?,
    _ error: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> UnsafeMutableRawPointer? {
    let v: AsrModelVersion = version == 2 ? .v2 : .v3
    let shared = AsrModels.defaultCacheDirectory(for: v)
    let dir: URL? =
        AsrModels.modelsExist(at: shared, version: v)
        ? shared : modelsDir.map { URL(fileURLWithPath: String(cString: $0)) }
    let progress = ProgressBox()
    let (result, _) = runBlocking(progress: progress, callback: callback, ctx: ctx) {
        () async throws -> AsrManager in
        let models = try await AsrModels.downloadAndLoad(
            to: dir, version: v,
            progressHandler: { p in
                switch p.phase {
                case .compiling: progress.set(fraction: p.fractionCompleted, text: "compile")
                default: progress.set(fraction: p.fractionCompleted, text: "download")
                }
            })
        let config = ASRConfig(
            tdtConfig: TdtConfig(blankId: v.blankId), encoderHiddenSize: v.encoderHiddenSize)
        let manager = AsrManager(config: config)
        try await manager.loadModels(models)
        return manager
    }
    switch result {
    case .success(let manager):
        return Unmanaged.passRetained(Engine(manager: manager)).toOpaque()
    case .failure(let e):
        setError(error, e)
        return nil
    }
}

/// Transcribe 16 kHz mono f32 PCM. Progress text is the text of each window finished so far,
/// separated by U+001E (windows overlap; the caller merges them).
/// Returns JSON:
///   {"cancelled": false, "words": [{"w": "Hello,", "s": 0.32, "e": 0.64}, ...]}
///   {"cancelled": true, "windows": [{"text": "...", "s": 0.0, "e": 14.9}, ...]}  (finished prefix)
/// Returns NULL and sets `error` on failure.
@_cdecl("ptb_transcribe")
public func ptb_transcribe(
    _ engine: UnsafeMutableRawPointer,
    _ samples: UnsafePointer<Float>,
    _ count: Int,
    _ callback: PtbCallback?,
    _ ctx: UnsafeMutableRawPointer?,
    _ error: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> UnsafeMutablePointer<CChar>? {
    let manager = Unmanaged<Engine>.fromOpaque(engine).takeUnretainedValue().manager
    let pcm = Array(UnsafeBufferPointer(start: samples, count: count))
    let progress = ProgressBox()
    let (result, cancelled) = runBlocking(progress: progress, callback: callback, ctx: ctx) {
        () async throws -> ASRResult in
        await manager.setWindowHandler { progress.addWindow($0) }
        var state = TdtDecoderState.make(decoderLayers: await manager.decoderLayerCount)
        defer { Task { await manager.setWindowHandler(nil) } }
        return try await manager.transcribe(pcm, decoderState: &state)
    }
    if cancelled {
        let windows = progress.completedWindows().map { ["text": $0.text, "s": $0.start, "e": $0.end] }
        return jsonString(["cancelled": true, "windows": windows])
    }
    switch result {
    case .success(let r):
        let words = buildWordTimings(from: r.tokenTimings ?? []).map {
            ["w": $0.word, "s": $0.startTime, "e": $0.endTime] as [String: Any]
        }
        // Without timings (shouldn't happen for TDT), fall back to the plain text.
        if words.isEmpty && !r.text.isEmpty {
            return jsonString(["cancelled": false, "words": [["w": r.text, "s": 0.0, "e": r.duration]]])
        }
        return jsonString(["cancelled": false, "words": words])
    case .failure(let e):
        setError(error, e)
        return nil
    }
}

@_cdecl("ptb_release")
public func ptb_release(_ engine: UnsafeMutableRawPointer) {
    Unmanaged<Engine>.fromOpaque(engine).release()
}

@_cdecl("ptb_free_string")
public func ptb_free_string(_ s: UnsafeMutablePointer<CChar>?) {
    free(s)
}
