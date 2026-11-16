import React from "react";
import { AbsoluteFill, Img, staticFile } from "remotion";
import { Backdrop } from "./Episode";
import { BRAND, LogoMark, Wordmark } from "./brand";
import { applyDesign } from "./design";
import { C } from "./theme";

// YouTube channel banner, 2560x1440. Only the centre 1546x423 shows on every device (phones crop to it), so the
// lockup, the channel line and the thumbnail cards all sit inside SAFE; the grid backdrop fills the rest for TV and
// desktop. Thumbnails are real episode thumbnails, passed as paths relative to --public-dir.
export type BannerProps = { thumbs: string[]; line?: string; sub?: string };

const W = 2560, H = 1440;
const SAFE = { w: 1546, h: 423 };
const SX = (W - SAFE.w) / 2, SY = (H - SAFE.h) / 2;

export const Banner: React.FC<BannerProps> = ({ thumbs, line = "Every week: what's new in robotics and physical AI", sub = "Research, products and deals · every source linked" }) => {
  applyDesign("blueprint-light");
  const cards = thumbs.slice(0, 3);
  const tilt = [-6, 3, -2];
  return (
    <AbsoluteFill style={{ fontFamily: "'Helvetica Neue', Helvetica, Arial, sans-serif" }}>
      <Backdrop kind="blueprint" />
      {/* Left half of the safe area: the mark, the name, the line. */}
      <div style={{ position: "absolute", left: SX + 10, top: SY, width: 690, height: SAFE.h, display: "flex", flexDirection: "column", justifyContent: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 24 }}>
          <LogoMark size={96} />
          <Wordmark size={82} style={{ color: C.ink }} />
        </div>
        <div style={{ marginTop: 30, fontSize: 42, fontWeight: 800, lineHeight: 1.15, color: C.ink, letterSpacing: -0.5 }}>{line}</div>
        <div style={{ marginTop: 14, fontSize: 26, fontWeight: 600, color: C.dim }}>{sub}</div>
        <div style={{ marginTop: 26, width: 140, height: 10, borderRadius: 5, background: BRAND.red }} />
      </div>
      {/* Right half: real episode thumbnails as slightly tilted cards. */}
      <div style={{ position: "absolute", left: SX + 720, top: SY, width: SAFE.w - 720, height: SAFE.h }}>
        {cards.map((src, i) => {
          const w = 500, h = 281;
          const left = [0, 160, 320][i], top = [-25, 135, 0][i];
          return (
            <div key={i} style={{ position: "absolute", left, top, width: w, height: h, transform: `rotate(${tilt[i]}deg)`, zIndex: i === 1 ? 3 : 1,
              borderRadius: 14, overflow: "hidden", background: "#fff", boxShadow: "0 22px 50px rgba(31,32,35,0.25), 0 0 0 6px #fff" }}>
              <Img src={staticFile(src)} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

export const BANNER = { width: W, height: H };
