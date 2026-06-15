import fs from 'node:fs';
import path from 'node:path';

const root = process.cwd();
const sourcePath = path.join(root, 'seedmind', 'visualization', 'fouloides_viewer.html');
const outDir = path.join(root, 'public');
const outPath = path.join(outDir, 'index.html');
const wsUrl = process.env.SEEDMIND_WS_URL || '';

let html = fs.readFileSync(sourcePath, 'utf8');
html = html.replace('"__SEEDMIND_WS_URL__"', JSON.stringify(wsUrl));

fs.mkdirSync(outDir, { recursive: true });
fs.writeFileSync(outPath, html);

console.log(`Built ${path.relative(root, outPath)}`);
if (wsUrl) {
  console.log(`Configured WebSocket backend: ${wsUrl}`);
} else {
  console.log('No SEEDMIND_WS_URL set; use ?ws=wss://... at runtime.');
}
