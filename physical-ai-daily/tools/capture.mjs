#!/usr/bin/env node
// Capture a source page as a video-ready screenshot, optionally highlighting a sentence in-page.
//
// Usage:
//   node tools/capture.mjs page  <url> <out.png> [--find "exact sentence"] [--w 1600] [--h 900]
//   node tools/capture.mjs tweet <tweet_id> <out.png>
//   node tools/capture.mjs element <url> <out.png> --selector "figure" [--nth 0]
//   node tools/capture.mjs frames  <url> <prefix.png> --selector "video" --nth 0 --times "2,6,10"
//
// Writes <out.png> plus <out>.json with {width,height,highlight:{x,y,w,h}|null} so the
// compositor can zoom onto the highlighted region.
import { chromium } from "../video/node_modules/playwright-core/index.mjs";
import fs from "node:fs";

const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const [mode, target, out, ...rest] = process.argv.slice(2);
const opt = {};
for (let i = 0; i < rest.length; i += 2) opt[rest[i].replace(/^--/, "")] = rest[i + 1];
const W = Number(opt.w || 1600), H = Number(opt.h || 900);

const browser = await chromium.launch({ executablePath: CHROME, headless: true });
const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 2 });
let meta = { width: W, height: H, highlight: null };

try {
  if (mode === "tweet") {
    await page.setViewportSize({ width: 600, height: 900 });
    await page.goto(`https://platform.twitter.com/embed/Tweet.html?id=${target}&theme=dark&dnt=true&hideThread=true`, { waitUntil: "networkidle" });
    const card = page.locator("article").first();
    await card.waitFor({ timeout: 15000 });
    await page.waitForTimeout(1500); // let media thumbnails decode
    await card.screenshot({ path: out });
    const box = await card.boundingBox();
    meta = { width: Math.round(box.width), height: Math.round(box.height), highlight: null };
  } else if (mode === "element") {
    // Screenshot one element (e.g. a paper figure): --selector "figure" [--nth 0]
    await page.goto(target, { waitUntil: "networkidle", timeout: 45000 });
    const el = page.locator(opt.selector).nth(Number(opt.nth || 0));
    await el.scrollIntoViewIfNeeded();
    await page.waitForTimeout(800);
    await el.screenshot({ path: out });
    const box = await el.boundingBox();
    meta = { width: Math.round(box.width), height: Math.round(box.height), highlight: null };
  } else if (mode === "frames") {
    // Still frames from a video playing on the source page: --selector "video" --nth 0 --times "2,6,10".
    // <out.png> is a prefix: writes <out>_<t>s.png (+ .json) per time.
    await page.goto(target, { waitUntil: "networkidle", timeout: 45000 }).catch(() => {});
    const el = page.locator(opt.selector || "video").nth(Number(opt.nth || 0));
    await el.scrollIntoViewIfNeeded();
    await el.evaluate((v) => { v.pause(); v.muted = true; });
    for (const t of (opt.times || "1").split(",").map(Number)) {
      await el.evaluate((v, t) => new Promise((res) => {
        const go = () => { v.addEventListener("seeked", () => res(), { once: true }); v.currentTime = Math.min(t, v.duration - 0.1); };
        v.readyState >= 1 ? go() : v.addEventListener("loadedmetadata", go, { once: true });
      }), t);
      await page.waitForTimeout(400);
      const file = out.replace(/\.png$/, `_${t}s.png`);
      await el.screenshot({ path: file });
      const box = await el.boundingBox();
      fs.writeFileSync(file.replace(/\.png$/, ".json"), JSON.stringify({ width: Math.round(box.width), height: Math.round(box.height), highlight: null }));
      console.log(`OK ${file}`);
    }
    meta = null;
  } else {
    await page.goto(target, { waitUntil: "domcontentloaded", timeout: 45000 });
    await page.waitForLoadState("networkidle", { timeout: 15000 }).catch(() => {});
    // Privacy-preserving consent handling: only ever click reject/necessary-only buttons.
    for (const re of [/reject all/i, /decline/i, /only necessary|necessary only|essential only/i]) {
      const b = page.getByRole("button", { name: re }).first();
      if (await b.isVisible().catch(() => false)) { await b.click().catch(() => {}); break; }
    }
    // Banners with no reject button (e.g. a fixed privacy notice): hide them without clicking anything.
    await page.evaluate(() => {
      for (const el of document.querySelectorAll("body *")) {
        const cs = getComputedStyle(el);
        if ((cs.position === "fixed" || cs.position === "sticky") && /privacy|cookie|consent/i.test(el.textContent || "") && el.offsetHeight < 600 && !/^(header|nav)$/i.test(el.tagName)) el.style.display = "none";
      }
    });
    if (opt.find) {
      const box = await page.evaluate((needle) => {
        const norm = (s) => s.replace(/\s+/g, " ");
        const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
        let node;
        while ((node = walker.nextNode())) {
          const raw = node.textContent;
          // Build the whitespace-collapsed text alongside a map from each normalized char to its raw offset.
          let text = "", map = [];
          for (let r = 0; r < raw.length; r++) {
            if (/\s/.test(raw[r]) && /\s/.test(raw[r + 1] || "")) continue;
            text += /\s/.test(raw[r]) ? " " : raw[r];
            map.push(r);
          }
          const i = text.indexOf(norm(needle));
          if (i < 0) continue;
          const range = document.createRange();
          range.setStart(node, map[i]);
          range.setEnd(node, map[i + norm(needle).length - 1] + 1);
          const mark = document.createElement("mark");
          mark.style.cssText = "background:#ffe14d;color:#111;padding:2px 0;border-radius:3px;box-shadow:0 0 0 3px #ffe14d";
          try { range.surroundContents(mark); } catch { return null; }
          mark.scrollIntoView({ block: "center" });
          const b = mark.getBoundingClientRect();
          return { x: b.x, y: b.y, w: b.width, h: b.height };
        }
        return null;
      }, opt.find);
      if (!box) console.error(`WARN: text not found on page: "${opt.find.slice(0, 60)}..."`);
      await page.waitForTimeout(1000);
      // Smooth-scrolling pages move after the first measurement: scroll instantly, settle, then re-measure.
      if (box) {
        await page.evaluate(() => document.querySelector("mark").scrollIntoView({ block: "center", behavior: "instant" }));
        await page.waitForTimeout(1500);
        meta.highlight = await page.evaluate(() => { const b = document.querySelector("mark").getBoundingClientRect(); return { x: b.x, y: b.y, w: b.width, h: b.height }; });
      } else meta.highlight = null;
    }
    await page.screenshot({ path: out });
  }
  if (meta) fs.writeFileSync(out.replace(/\.png$/, ".json"), JSON.stringify(meta, null, 2));
  if (meta) console.log(`OK ${out} ${JSON.stringify(meta)}`);
} finally {
  await browser.close();
}
