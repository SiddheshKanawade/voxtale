import React from "react";
import { AbsoluteFill, Img, staticFile } from "remotion";
import { C, FONT } from "./theme";
import { BRAND, LogoMark } from "./brand";
import { applyDesign } from "./design";
import { Backdrop } from "./Episode";

// YouTube thumbnail, 1280x720 (docs/thumbnails.md): one focal image, at most ~4 big words, high contrast, readable at
// 168x94. Same design tokens as the episode, plus a small brand mark in the same corner every time (series cue),
// top-left because YouTube covers the bottom-right with the duration.
// Layouts: "face" = image on the right, words on the left; "stat" = one huge number over a faded image;
// "split" = two images side by side under a short line.
export type ThumbProps = {
  design?: string;
  layout: "face" | "stat" | "split";
  lines: string[];        // 1-2 short lines; wrap a word in *asterisks* to colour it
  image?: string;         // episode-relative path
  image2?: string;        // second image, for "split"
  focus?: string;         // CSS object-position for the image crop, e.g. "70% 30%"
  stat?: string;          // the big number, for "stat"
  label?: string;         // small chip above the words, e.g. "NEWS"
};

const words = (line: string, accent: string) =>
  line.split(/(\*[^*]+\*)/).map((part, i) =>
    part.startsWith("*") ? <span key={i} style={{ color: accent }}>{part.slice(1, -1)}</span> : <span key={i}>{part}</span>);

export const Thumbnail: React.FC<ThumbProps> = ({ design, layout, lines, image, image2, focus = "50% 50%", stat, label }) => {
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
