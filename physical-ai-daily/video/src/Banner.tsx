import React from "react";
import { AbsoluteFill, Img, staticFile } from "remotion";
import { Backdrop } from "./Episode";
import { BRAND, LogoMark, Wordmark } from "./brand";
import { applyDesign } from "./design";
import { C } from "./theme";

// YouTube channel banner, 2560x1440, designed for YouTube's three crops of the one image:
// - every device sees the centre 1546x423 (SAFE): the logo, name and channel line live here, centred;
// - desktop sees the full-width band 2560x423: a thumbnail card in each side wing completes the design there
//   (phones crop the wings away, which only removes decoration);
// - TV sees all of it: the grid backdrop carries on above and below the band.
// Thumbnails are real episode thumbnails, passed as paths relative to --public-dir.
export type BannerProps = { thumbs: string[]; line?: string; sub?: string };

const W = 2560, H = 1440;
const SAFE = { w: 1546, h: 423 };
const SX = (W - SAFE.w) / 2, BY = (H - SAFE.h) / 2; // safe area left edge; band top edge (band = full width, SAFE.h tall)
const CARD = { w: 600, h: 338 };                     // 16:9; reaches past the 507 px wing into the safe area's empty sides

export const Banner: React.FC<BannerProps> = ({ thumbs, line = "Every week: what's new in robotics and physical AI", sub = "Research, products and deals · every source linked" }) => {
  applyDesign("blueprint-light");
  const wings = [thumbs[0], thumbs[1]].filter(Boolean) as string[];
  return (
    <AbsoluteFill style={{ fontFamily: "'Helvetica Neue', Helvetica, Arial, sans-serif" }}>
      <Backdrop kind="blueprint" />
      {/* Everyone: the essentials, centred in the safe area. */}
      <div style={{ position: "absolute", left: SX, top: BY, width: SAFE.w, height: SAFE.h, display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center", textAlign: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 28 }}>
          <LogoMark size={118} />
          <Wordmark size={100} style={{ color: C.ink }} />
        </div>
        <div style={{ marginTop: 30, fontSize: 50, fontWeight: 800, color: C.ink, letterSpacing: -0.5 }}>{line}</div>
        <div style={{ marginTop: 14, fontSize: 30, fontWeight: 600, color: C.dim }}>{sub}</div>
        <div style={{ marginTop: 24, width: 140, height: 10, borderRadius: 5, background: BRAND.red }} />
      </div>
      {/* Desktop: one real episode thumbnail in each side wing of the band, tilted toward the centre. */}
      {wings.map((src, i) => {
        const left = i === 0 ? 34 : W - 34 - CARD.w;
        return (
          <div key={i} style={{ position: "absolute", left, top: BY + (SAFE.h - CARD.h) / 2, width: CARD.w, height: CARD.h,
            transform: `rotate(${i === 0 ? -4 : 4}deg)`, borderRadius: 14, overflow: "hidden", background: "#fff",
            boxShadow: "0 22px 50px rgba(31,32,35,0.25), 0 0 0 6px #fff" }}>
            <Img src={staticFile(src)} style={{ width: "100%", height: "100%", objectFit: "cover" }} />
          </div>
        );
      })}
    </AbsoluteFill>
  );
};

export const BANNER = { width: W, height: H };
