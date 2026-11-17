import React from "react";
import { AbsoluteFill, Img, staticFile } from "remotion";
import { Backdrop } from "../Episode";
import { BRAND, Lockup } from "../brand";
import { applyDesign, current } from "../design";
import { C, FONT } from "../theme";

// Cover (thumbnail) for a Short, 1080x1920: a curiosity hook in big type over one or two stacked frames that show
// the story (e.g. "shown once" -> "does it alone"). Not the Short's first frame: the cover's job is to make someone
// tap. Hook lines take *asterisks* for the accent colour. Frames are paths relative to --public-dir.
export type CoverProps = {
  design?: string;
  lines: string[];
  panels: { src: string; label?: string }[];
  credit?: string;
};

const W = 1080, PAD = 56;

// Three lines must fit above the panels (y < 540): longer lines get a smaller size so a model name fits.
const hookSize = (lines: string[]) => {
  const n = Math.max(...lines.map((l) => l.replace(/\*/g, "").length));
  return n <= 16 ? 112 : n <= 19 ? 100 : 88;
};

const Accent: React.FC<{ text: string }> = ({ text }) => (
  <>{text.split("*").map((p, i) => <span key={i} style={i % 2 ? { color: current.light ? BRAND.red : BRAND.yellow } : undefined}>{p}</span>)}</>
);

export const Cover: React.FC<CoverProps> = ({ design, lines, panels, credit }) => {
  applyDesign(design);
  const n = Math.min(2, panels.length);
  const top = 560, gap = 34, ph = n === 2 ? 560 : 1000;
  return (
    <AbsoluteFill style={{ fontFamily: FONT, color: C.ink }}>
      <Backdrop kind="blueprint" />
      <div style={{ position: "absolute", left: PAD, top: 70 }}><Lockup size={46} /></div>
      <div style={{ position: "absolute", left: PAD, right: PAD, top: 170, fontSize: hookSize(lines), fontWeight: 900, lineHeight: 1.0, letterSpacing: -3 }}>
        {lines.map((l, i) => <div key={i}><Accent text={l} /></div>)}
      </div>
      {panels.slice(0, n).map((p, i) => (
        <div key={i} style={{ position: "absolute", left: PAD, width: W - 2 * PAD, top: top + i * (ph + gap), height: ph,
          borderRadius: 22, overflow: "hidden", boxShadow: current.light ? "0 18px 50px rgba(31,32,35,0.2)" : "0 24px 60px rgba(0,0,0,0.6)" }}>
          <Img src={staticFile(p.src)} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
          {p.label && (
            <div style={{ position: "absolute", left: 22, top: 22, background: i === n - 1 ? BRAND.red : "rgba(15,15,17,0.82)", color: "#fff",
              fontWeight: 900, fontSize: 46, padding: "10px 22px", borderRadius: 12, letterSpacing: 0.5 }}>{p.label}</div>
          )}
        </div>
      ))}
      {n === 2 && (
        <div style={{ position: "absolute", left: W / 2 - 42, top: top + ph + gap / 2 - 42, width: 84, height: 84, borderRadius: 42,
          background: BRAND.red, color: "#fff", fontSize: 52, fontWeight: 900, display: "flex", alignItems: "center", justifyContent: "center",
          boxShadow: "0 10px 30px rgba(0,0,0,0.4)", zIndex: 2 }}>↓</div>
      )}
      {credit && <div style={{ position: "absolute", left: PAD, bottom: 52, fontSize: 24, color: C.dim }}>{credit}</div>}
    </AbsoluteFill>
  );
};
