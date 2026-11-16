#!/usr/bin/env node
// Capture a page and highlight a sentence that spans several elements (links, <em>, MathJax), where
// capture.mjs --find fails. A DOM Range over the text nodes; instant scroll, settle, re-measure.
// Usage: node tools/highlight.mjs <url> <out.png> "<exact text>" [--w 1600] [--wait 2500] [--hideads 1]
import { chromium } from "../video/node_modules/playwright-core/index.mjs";
import fs from "node:fs";
const [url, out, needle, ...rest] = process.argv.slice(2);
const opt = {}; for (let i = 0; i < rest.length; i += 2) opt[rest[i].replace(/^--/, "")] = rest[i + 1];
const b = await chromium.launch({ executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless: true });
const VW = Number(opt.w || 1600), VH = Math.round(VW * 9 / 16);
const p = await b.newPage({ viewport: { width: VW, height: VH }, deviceScaleFactor: 3200 / VW });
try {
  await p.goto(url, { waitUntil: "domcontentloaded", timeout: 45000 });
  await p.waitForLoadState("networkidle", { timeout: 15000 }).catch(() => {});
  for (const re of [/reject all/i, /decline/i, /only necessary|necessary only|essential only/i]) {
    const x = p.getByRole("button", { name: re }).first();
    if (await x.isVisible().catch(() => false)) { await x.click().catch(() => {}); break; }
  }
  await p.evaluate(() => { document.documentElement.style.scrollBehavior = "auto"; document.body.style.scrollBehavior = "auto"; }).catch(() => {});
  if (opt.hideads) await p.evaluate(() => {
    const kill = (e) => { e.style.setProperty("display", "none", "important"); };
    document.querySelectorAll('iframe[src*="doubleclick"], iframe[src*="googlesyndication"], iframe[id^="google_ads"], ins.adsbygoogle, [id^="google_ads"], [class*="adsbygoogle"], [class*="ad-container"], [id*="div-gpt"]').forEach(kill);
    document.querySelectorAll("iframe").forEach((f) => { if (!/twitter|youtube/.test(f.src)) kill(f.parentElement && f.parentElement.children.length === 1 ? f.parentElement : f); });
  });
  const found = await p.evaluate((needle) => {
    const norm = (s) => s.replace(/\s+/g, " ");
    const nodes = []; const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT); let n;
    let text = "", map = [];
    while ((n = w.nextNode())) {
      const el = n.parentElement; if (!el || ["SCRIPT","STYLE","NOSCRIPT"].includes(el.tagName)) continue;
      const raw = n.textContent;
      for (let r = 0; r < raw.length; r++) {
        const ch = /\s/.test(raw[r]) ? " " : raw[r];
        if (ch === " " && text.endsWith(" ")) continue;
        text += ch; map.push([n, r]);
      }
    }
    const nd = norm(needle); let i = text.indexOf(nd);
    if (i < 0) return false;
    const range = document.createRange();
    range.setStart(map[i][0], map[i][1]);
    const e = map[i + nd.length - 1]; range.setEnd(e[0], e[1] + 1);
    window.__r = range;
    const r = range.getBoundingClientRect();
    window.scrollTo({ top: window.scrollY + r.top - innerHeight / 2 + r.height / 2, behavior: "instant" });
    return true;
  }, needle);
  if (!found) { console.error(`WARN: text not found on page: "${needle.slice(0, 60)}..."`); process.exitCode = 2; }
  await p.waitForTimeout(Number(opt.wait || 2500));
  const box = found ? await p.evaluate(() => {
    const r = window.__r; const rb = r.getBoundingClientRect();
    window.scrollTo({ top: window.scrollY + rb.top - innerHeight / 2 + rb.height / 2, behavior: "instant" });
    const rects = [...r.getClientRects()].filter((q) => q.width > 1);
    for (const q of rects) {
      const d = document.createElement("div");
      d.style.cssText = `position:fixed;left:${q.left - 2}px;top:${q.top - 1}px;width:${q.width + 4}px;height:${q.height + 2}px;background:#ffe14d;mix-blend-mode:multiply;border-radius:3px;z-index:2147483647;pointer-events:none`;
      document.body.appendChild(d);
    }
    const f = r.getBoundingClientRect();
    return { x: f.x, y: f.y, w: f.width, h: f.height };
  }) : null;
  await p.waitForTimeout(500);
  await p.screenshot({ path: out });
  const meta = { width: VW, height: VH, highlight: box };
  fs.writeFileSync(out.replace(/\.png$/, ".json"), JSON.stringify(meta, null, 2));
  console.log(`OK ${out} ${JSON.stringify(meta)}`);
} finally { await b.close(); }
