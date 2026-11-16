#!/usr/bin/env node
// Usage: node tools/video_frames.mjs <url> <prefix.png> <nth video> <times "2,6,10">
// Video stills with the <video> lifted full-viewport above overlays (play buttons, cookie banners). Same output as capture.mjs frames.
import { chromium } from "../video/node_modules/playwright-core/index.mjs";
import fs from "node:fs";
const [url, out, nth, times] = process.argv.slice(2);
const b = await chromium.launch({ executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless: true });
const p = await b.newPage({ viewport: { width: 1600, height: 900 }, deviceScaleFactor: 1.2 });
await p.goto(url, { waitUntil: "networkidle", timeout: 45000 }).catch(() => {});
for (const re of [/reject all/i, /decline/i, /only necessary|necessary only|essential only/i]) {
  const x = p.getByRole("button", { name: re }).first();
  if (await x.isVisible().catch(() => false)) { await x.click().catch(() => {}); break; }
}
const el = p.locator("video").nth(Number(nth));
await el.scrollIntoViewIfNeeded(); await p.waitForTimeout(1500);
const ar = await el.evaluate((v) => new Promise((res) => {
  const go = () => { v.pause(); v.muted = true;
    v.style.cssText = "position:fixed;left:0;top:0;width:100vw;height:100vh;object-fit:contain;background:#000;z-index:2147483647;opacity:1;transform:none;filter:none";
    res(v.videoWidth / v.videoHeight); };
  v.readyState >= 1 ? go() : (v.addEventListener("loadedmetadata", go, { once: true }), v.load());
}));
const W = 1600, H = Math.round(1600 / ar);
await p.setViewportSize({ width: W, height: H });
for (const t of times.split(",").map(Number)) {
  await el.evaluate((v, t) => new Promise((res) => { v.addEventListener("seeked", () => res(), { once: true }); v.currentTime = Math.min(t, v.duration - 0.1); }), t);
  await p.waitForTimeout(500);
  const file = out.replace(/\.png$/, `_${t}s.png`);
  await p.screenshot({ path: file });
  fs.writeFileSync(file.replace(/\.png$/, ".json"), JSON.stringify({ width: W, height: H, highlight: null }));
  console.log("OK", file);
}
await b.close();
