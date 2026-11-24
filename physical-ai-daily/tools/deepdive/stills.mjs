#!/usr/bin/env node
// Render chosen frames of a deep dive as PNGs in one or more designs, bundling once and reusing one browser (a
// `remotion still` per frame re-bundles and relaunches Chrome each time). props.json is read, never written: the design
// is overridden in the input props. Called by tools/deepdive/stills.sh.
// Usage: node tools/deepdive/stills.mjs <dd-dir> <out-dir> <design[,design]> <scale> <frame>...
import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const VIDEO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../video");
const require = createRequire(path.join(VIDEO, "package.json"));
const { bundle } = require("@remotion/bundler");
const { openBrowser, renderStill, selectComposition } = require("@remotion/renderer");

const [dir, out, designs, scale, ...frames] = process.argv.slice(2);
if (!frames.length) { console.error("usage: stills.mjs <dd-dir> <out-dir> <design[,design]> <scale> <frame>..."); process.exit(2); }
const props = JSON.parse(fs.readFileSync(path.join(dir, "props.json"), "utf8"));
const NAME = { "blueprint-light": "light", blueprint: "dark" };
fs.mkdirSync(out, { recursive: true });

const serveUrl = await bundle({ entryPoint: path.join(VIDEO, "src/deepdive/index.ts"), publicDir: path.resolve(dir), onProgress: () => {} });
const browser = await openBrowser("chrome");
try {
  for (const design of designs.split(",")) {
    const inputProps = { ...props, design };
    const composition = await selectComposition({ serveUrl, id: "DeepDive", inputProps, puppeteerInstance: browser });
    for (const f of frames.map(Number)) {
      const frame = Math.min(f, composition.durationInFrames - 1);
      const file = path.join(out, `${NAME[design] ?? design}-f${String(frame).padStart(5, "0")}.png`);
      await renderStill({ composition, serveUrl, output: file, frame, inputProps, puppeteerInstance: browser, scale: Number(scale) });
      console.log(file);
    }
  }
} finally {
  await browser.close({ silent: true });
  fs.rmSync(serveUrl, { recursive: true, force: true }); // the bundle holds a copy of the episode's media
}
