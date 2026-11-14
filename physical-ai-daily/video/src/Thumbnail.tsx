import React from "react";
import { AbsoluteFill, Img, staticFile } from "remotion";
import { C, FONT } from "./theme";
import { BRAND, LogoMark } from "./brand";
import { applyDesign } from "./design";
import { Backdrop } from "./Episode";

// YouTube thumbnail, 1280x720 (docs/thumbnails.md): one focal image, at most ~4 big words, high contrast, readable at
// 168x94. Same design tokens as the episode, plus a small brand mark in the same corner every time (series cue),
// top-left because YouTube covers the bottom-right with the duration.
// Layouts: "evidence" (default) = subject on the right, a hook on the left and, under it, a tilted "receipt": a crop of
// the source with the key claim highlighted plus a source tag (source cues raise clicks and trust; docs/thumbnails.md).
// "face" = image on the right, words on the left; "stat" = one huge number over a faded image;
// "split" = two images side by side under a short line.
export type ThumbProps = {
  design?: string;
  layout: "evidence" | "face" | "stat" | "split";
  lines: string[];        // 1-2 short lines; wrap a word in *asterisks* to colour it
  image?: string;         // episode-relative path
  image2?: string;        // second image, for "split"
  focus?: string;         // CSS object-position for the image crop, e.g. "70% 30%"
  zoom?: number;          // scale the subject image up to crop away figure labels or borders (default 1)
  stat?: string;          // the big number, for "stat"
  label?: string;         // small chip above the words, e.g. "NEWS"
  receipt?: Receipt;      // "evidence" layout: the proof card
};

// A crop of a captured source page. Fractions are of the full image; mark (optional) is a highlighter box, in fractions
// of the crop, for pages captured without one. aspect = image height / width (tools/thumbnails.py fills it in).
export type Receipt = {
  src: string; source: string; aspect?: number;
  crop: { x: number; y: number; w: number; h: number };
  mark?: { x: number; y: number; w: number; h: number };
};

const RECEIPT_W = 560;
const ReceiptCard: React.FC<{ r: Receipt; light?: boolean }> = ({ r, light }) => {
  const imgW = RECEIPT_W / r.crop.w;
  const imgH = imgW * (r.aspect ?? 0.5625);
  const h = r.crop.h * imgH;
  return (
    <div style={{ transform: "rotate(-3deg)", transformOrigin: "left top", width: RECEIPT_W }}>
      <div style={{ display: "inline-block", background: light ? "#1f2023" : "#ffffff", color: light ? "#fff" : "#1f2023", fontFamily: FONT,
        fontWeight: 800, fontSize: 22, letterSpacing: 2, padding: "5px 12px", borderRadius: "8px 8px 0 0" }}>SOURCE: {r.source}</div>
      <div style={{ position: "relative", width: RECEIPT_W, height: h, overflow: "hidden", borderRadius: "0 10px 10px 10px",
        border: `4px solid ${light ? "#1f2023" : "#ffffff"}`, boxShadow: "0 20px 50px rgba(0,0,0,0.45)" }}>
        <Img src={staticFile(r.src)} style={{ position: "absolute", width: imgW, height: imgH, left: -r.crop.x * imgW, top: -r.crop.y * imgH }} />
        {r.mark && <div style={{ position: "absolute", left: r.mark.x * RECEIPT_W, top: r.mark.y * h, width: r.mark.w * RECEIPT_W,
          height: r.mark.h * h, background: "rgba(240,201,69,0.45)", mixBlendMode: "multiply", borderRadius: 4,
          boxShadow: `0 0 0 3px ${BRAND.red}` }} />}
      </div>
    </div>
  );
};

const words = (line: string, accent: string) =>
  line.split(/(\*[^*]+\*)/).map((part, i) =>
    part.startsWith("*") ? <span key={i} style={{ color: accent }}>{part.slice(1, -1)}</span> : <span key={i}>{part}</span>);

export const Thumbnail: React.FC<ThumbProps> = ({ design, layout, lines, image, image2, focus = "50% 50%", zoom = 1, stat, label, receipt }) => {
  const look = applyDesign(design);
  const ink = look.light ? C.ink : "#ffffff";
  const accent = look.light ? BRAND.red : BRAND.yellow;
  const fadeTo = `rgba(${C.bgRgb},1)`;
  // Shrink the words until the longest line fits its box (heavy caps run about 0.68 em per character), so a line never
  // wraps: every line stays exactly as written.
  const fit = (size: number, width: number) =>
    Math.min(size, Math.floor(width / (0.68 * Math.max(...lines.map((l) => l.replace(/\*/g, "").length)))));
  const text = (size: number, style?: React.CSSProperties) => (
    <div style={{ fontFamily: FONT, fontWeight: 900, fontSize: size, whiteSpace: "nowrap", lineHeight: 0.95, letterSpacing: -size * 0.035,
      color: ink, textTransform: "uppercase", ...style }}>
      {lines.map((l, i) => <div key={i}>{words(l, accent)}</div>)}
    </div>
  );
  const chip = label && (
    <div style={{ display: "inline-block", background: BRAND.red, color: "#fff", fontFamily: FONT, fontWeight: 800, fontSize: 30,
      letterSpacing: 3, padding: "6px 16px", borderRadius: 8, marginBottom: 22 }}>{label}</div>
  );
  return (
    <AbsoluteFill style={{ background: C.bg, overflow: "hidden" }}>
      <Backdrop kind={look.backdrop} />
      {layout === "evidence" && image && (
        <>
          <div style={{ position: "absolute", right: 0, top: 0, width: 700, height: 720, overflow: "hidden" }}>
            <Img src={staticFile(image)} style={{ width: "100%", height: "100%", objectFit: "cover", objectPosition: focus,
              transform: `scale(${zoom})`, transformOrigin: focus }} />
          </div>
          <AbsoluteFill style={{ background: `linear-gradient(90deg, ${fadeTo} 44%, rgba(${C.bgRgb},0) 64%)` }} />
          <div style={{ position: "absolute", left: 56, top: 104, width: 680 }}>{text(fit(112, 680))}</div>
          {receipt && <div style={{ position: "absolute", left: 64, top: 372 }}><ReceiptCard r={receipt} light={look.light} /></div>}
        </>
      )}
      {layout === "face" && image && (
        <>
          <Img src={staticFile(image)} style={{ position: "absolute", right: 0, top: 0, width: 760, height: 720, objectFit: "cover", objectPosition: focus }} />
          <AbsoluteFill style={{ background: `linear-gradient(90deg, ${fadeTo} 38%, rgba(${C.bgRgb},0) 62%)` }} />
          <div style={{ position: "absolute", left: 64, top: 0, bottom: 0, width: 620, display: "flex", flexDirection: "column", justifyContent: "center" }}>
            <div>{chip}</div>
            {text(fit(118, 620))}
          </div>
        </>
      )}
      {layout === "stat" && (
        <>
          {image && <Img src={staticFile(image)} style={{ position: "absolute", inset: 0, width: 1280, height: 720, objectFit: "cover", objectPosition: focus, opacity: look.light ? 0.5 : 0.45 }} />}
          <AbsoluteFill style={{ background: `radial-gradient(ellipse 70% 70% at 35% 55%, ${fadeTo} 20%, rgba(${C.bgRgb},0.55) 100%)` }} />
          <div style={{ position: "absolute", left: 64, top: 0, bottom: 0, display: "flex", flexDirection: "column", justifyContent: "center" }}>
            <div>{chip}</div>
            <div style={{ fontFamily: FONT, fontWeight: 900, fontSize: 300, lineHeight: 0.85, letterSpacing: -14, color: accent }}>{stat}</div>
            {text(fit(84, 1100), { marginTop: 18 })}
          </div>
        </>
      )}
      {layout === "split" && image && image2 && (
        <>
          {[image, image2].map((src, i) => (
            <Img key={src} src={staticFile(src)} style={{ position: "absolute", top: 250, left: i ? 652 : 0, width: 628, height: 470, objectFit: "cover",
              objectPosition: i ? "50% 60%" : focus }} />
          ))}
          <div style={{ position: "absolute", left: 640, top: 250, bottom: 0, width: 12, background: BRAND.red }} />
          <div style={{ position: "absolute", left: 112, right: 40, top: 40 }}>{text(fit(104, 1120))}</div>
        </>
      )}
      {/* top-left: YouTube draws the duration over the bottom-right corner */}
      <div style={{ position: "absolute", left: 28, top: 28 }}><LogoMark size={56} /></div>
    </AbsoluteFill>
  );
};
