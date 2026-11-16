import React from "react";
import { AbsoluteFill, Easing, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { C } from "./theme";

// Pixels2Actions brand (pixelstoactions.com): the mark is a red tile holding a 2x2 grid of pixels (one yellow)
// and an arrow, read left to right as "pixels -> actions". Colors and geometry follow the site's logo-mark.svg.
export const BRAND = { red: "#f43f2e", yellow: "#f0c945", name: ["Pixels", "2", "Actions"] as const };
export const WORDMARK_FONT = "'Helvetica Neue', Helvetica, Arial, sans-serif";

const PIXELS = [{ x: 11, y: 17, fill: BRAND.yellow }, { x: 26, y: 17, fill: "#fff" }, { x: 11, y: 32, fill: "#fff" }, { x: 26, y: 32, fill: "#fff" }];

// pixels: 0..1 per pixel pop-in (staggered), arrow: 0..1 slide-in. Both default to fully drawn.
export const LogoMark: React.FC<{ size: number; pixels?: number[]; arrow?: number }> = ({ size, pixels = [1, 1, 1, 1], arrow = 1 }) => (
  <svg width={size} height={size} viewBox="0 0 64 64" style={{ display: "block", flexShrink: 0 }}>
    <rect width="64" height="64" rx="15" fill={BRAND.red} />
    {PIXELS.map((p, i) => (
      <rect key={i} x={p.x} y={p.y} width="11" height="11" rx="2.2" fill={p.fill}
        transform={`translate(${p.x + 5.5} ${p.y + 5.5}) scale(${pixels[i]}) translate(${-p.x - 5.5} ${-p.y - 5.5})`} />
    ))}
    <path d="M40 15.5 L56.5 32 L40 48.5 Z" fill="#fff" opacity={arrow} transform={`translate(${(1 - arrow) * -14} 0)`} />
  </svg>
);

// Channel profile picture: YouTube crops it to a circle, so the tile's red fills the whole square and the pixels and
// arrow sit centred (their bounding box is right of centre in the 64-unit tile) with room inside the circle.
export const ChannelIcon: React.FC = () => (
  <svg width="100%" height="100%" viewBox="0 0 64 64" style={{ display: "block" }}>
    <rect width="64" height="64" fill={BRAND.red} />
    <g transform="translate(32 32) scale(0.82) translate(-33.75 -32)">
      {PIXELS.map((p, i) => <rect key={i} x={p.x} y={p.y} width="11" height="11" rx="2.2" fill={p.fill} />)}
      <path d="M40 15.5 L56.5 32 L40 48.5 Z" fill="#fff" />
    </g>
  </svg>
);

export const Wordmark: React.FC<{ size: number; style?: React.CSSProperties }> = ({ size, style }) => (
  <span style={{ fontFamily: WORDMARK_FONT, fontWeight: 700, fontSize: size, letterSpacing: -size * 0.03, color: C.ink, ...style }}>
    {BRAND.name[0]}<span style={{ color: BRAND.red }}>{BRAND.name[1]}</span>{BRAND.name[2]}
  </span>
);

export const Lockup: React.FC<{ size: number }> = ({ size }) => (
  <span style={{ display: "inline-flex", alignItems: "center", gap: size * 0.32 }}>
    <LogoMark size={size * 1.15} />
    <Wordmark size={size} />
  </span>
);

// Short sting between the cold open and the first story, over the sonic logo: the pixels pop in one by one, then the arrow
// drives forward and the name wipes in.
export const STING_S = 1.6;
export const END_S = 1.6; // end card hold after the sign-off, for the sonic logo

export const Sting: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const pixels = [0, 1, 2, 3].map((i) => spring({ frame: frame - 2 - i * 3, fps, config: { damping: 12, mass: 0.5 } }));
  const arrow = interpolate(frame, [15, 23], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.cubic) });
  const wipe = interpolate(frame, [19, 31], [0, 100], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.cubic) });
  const tile = spring({ frame, fps, config: { damping: 14 } });
  const out = interpolate(frame, [durationInFrames - 8, durationInFrames], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  return (
    <AbsoluteFill style={{ background: C.bg, justifyContent: "center", alignItems: "center", opacity: out }}>
      <div style={{ display: "flex", alignItems: "center", gap: 44 }}>
        <div style={{ transform: `scale(${0.6 + 0.4 * tile})`, opacity: tile }}>
          <LogoMark size={190} pixels={pixels} arrow={arrow} />
        </div>
        <div style={{ clipPath: `inset(0 ${100 - wipe}% 0 0)` }}>
          <Wordmark size={132} />
        </div>
      </div>
    </AbsoluteFill>
  );
};
