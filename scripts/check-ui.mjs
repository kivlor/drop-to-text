// Syntax-checks the inline <script> in ui/index.html (there's no bundler to do it for us).
import { readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const html = readFileSync(new URL('../ui/index.html', import.meta.url), 'utf8');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]);
if (!scripts.length) throw new Error('no inline <script> found in ui/index.html');

const dir = mkdtempSync(join(tmpdir(), 'check-ui-'));
scripts.forEach((src, i) => {
  const file = join(dir, `script-${i}.js`);
  writeFileSync(file, src);
  execFileSync(process.execPath, ['--check', file], { stdio: 'inherit' });
});
console.log(`ui/index.html: ${scripts.length} script block(s) OK`);
