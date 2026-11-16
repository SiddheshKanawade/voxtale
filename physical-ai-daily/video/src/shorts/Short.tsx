import React from "react";
import { AbsoluteFill, Audio, Img, OffthreadVideo, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { Backdrop } from "../Episode";
import { BRAND, Lockup } from "../brand";
import { applyDesign, current } from "../design";
import { Beat, C, FONT, Media, Word } from "../theme";

// A 9:16 promotional Short: one story, a hook on the first frame, word-synced captions, the payoff, then the way to
// the full episode. Props are a short's props.json from tools/voice.py (one "story" segment), plus top-level
// "hook" (on-screen line from frame 0) and "cta" (end-card line). Layout keeps text out of YouTube's Shorts UI:
// the right edge (buttons) and the bottom ~22% (title, channel, related-video link). See the promo-shorts skill.
export type ShortProps = {
  date: string; design?: string; hook: string; cta?: string; headline?: string; label?: string;
  segments: { id: string; beats: Beat[]; duration: number }[];
};

const W = 1080;
const SAFE = { left: 64, right: 150, top: 150, bottom: 430 }; // px kept clear of the Shorts UI
const MEDIA = { top: 440, height: 940 }; // the picture band (ends at 1380; captions under it, clear of the bottom UI)
export const END_S = 2.2; // end card hold after the last word

export const Short: React.FC<ShortProps> = ({ hook, cta, headline, label, segments, design }) => {
  applyDesign(design);
  const { fps } = useVideoConfig();
  const beats = segments.flatMap((s) => s.beats);
  let t = 0;
  const starts = beats.map((b) => { const at = t; t += b.duration; return at; });
  const total = t;
  return (
    <AbsoluteFill style={{ fontFamily: FONT, color: C.ink, overflow: "hidden" }}>
      <Backdrop kind="blueprint" />
      {beats.map((b, i) => (
        <Sequence key={i} from={Math.round(starts[i] * fps)} durationInFrames={Math.round(b.duration * fps)}>
          <ShortScene beat={b} first={i === 0} />
          <Audio src={staticFile(b.audio)} />
          <Captions words={b.words} />
        </Sequence>
      ))}
      <Sequence from={Math.round(total * fps)}>
        <EndCard cta={cta ?? "The full story is in this week's episode: tap the link below"} />
      </Sequence>
      <TopBar headline={headline} label={label} totalS={total + END_S} />
      <Hook text={hook} untilS={hookUntil(beats, starts, total)} />
    </AbsoluteFill>
  );
};

// Top: progress bar, lockup, and the story label. Small, so the picture and hook dominate.
const TopBar: React.FC<{ headline?: string; label?: string; totalS: number }> = ({ label, totalS }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return (
    <>
      <div style={{ position: "absolute", left: 0, top: 0, height: 10, width: `${Math.min(1, frame / (totalS * fps)) * 100}%`, background: BRAND.red }} />
      <div style={{ position: "absolute", left: SAFE.left, top: 70, display: "flex", alignItems: "center", gap: 18 }}>
        <Lockup size={44} />
        {label && (
          <span style={{ background: BRAND.red, color: "#fff", fontWeight: 800, fontSize: 26, padding: "6px 14px", borderRadius: 8, letterSpacing: 1 }}>
            {label}
          </span>
        )}
      </div>
    </>
  );
};

// The opening is two sentences (background, then the question) and may span two beats: keep the on-screen hook up
// until the beat that asks the question ends (the first beat ending in "?" among the first two), else the first beat.
const hookUntil = (beats: Beat[], starts: number[], total: number) => {
  const q = beats.slice(0, 2).findIndex((b) => b.narration.trim().endsWith("?"));
  const k = q >= 0 ? q + 1 : 1;
  return starts[k] ?? total;
};

// The hook: on screen from frame 0 (it doubles as the cover in the Shorts grid), through the first beat.
const Hook: React.FC<{ text: string; untilS: number }> = ({ text, untilS }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  if (frame > untilS * fps) return null;
  const out = interpolate(frame, [untilS * fps - 8, untilS * fps], [1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  // Fit in three lines above the picture band (~250 px): longer hooks get a smaller size, short ones shout.
  const chars = text.replace(/\*/g, "").length;
  const size = chars <= 36 ? 84 : chars <= 52 ? 74 : 64;
  return (
    <div style={{ position: "absolute", left: SAFE.left, right: SAFE.right, top: 160, opacity: out }}>
      <div style={{ fontSize: size, fontWeight: 900, lineHeight: 1.04, letterSpacing: -1.5, color: C.ink }}>
        {text.split("*").map((part, i) => (
          <span key={i} style={i % 2 ? { color: current.light ? BRAND.red : BRAND.yellow } : undefined}>{part}</span>
        ))}
      </div>
    </div>
  );
};

const ShortScene: React.FC<{ beat: Beat; first: boolean }> = ({ beat }) => {
  const v = beat.visual;
  if (v.type === "kinetic") return <Kinetic text={v.text ?? ""} sub={v.sub} />;
  if (v.type === "stat") return <Stat items={v.items ?? []} />;
  if (v.type === "split" || v.type === "panels") return <Stack items={v.items ?? []} credit={v.credit} />;
  if (v.type === "clip" && v.src) return <ClipBand m={v.src} credit={v.credit} stamp={v.stamp} />;
  if (v.src) return <Picture m={v.src} highlight={v.type === "shot"} credit={v.credit} stamp={v.stamp} badge={v.badge} fit={v.fit} />;
  return <Kinetic text={v.overlay ?? v.text ?? ""} />;
};

// A picture in the media band, with a slow push-in. For "shot" with a highlight, it frames the highlighted sentence.
const Picture: React.FC<{ m: Media; highlight?: boolean; credit?: string; stamp?: string; badge?: string; fit?: "cover" | "contain" }> = ({ m, highlight, credit, stamp, badge, fit }) => {
  const frame = useCurrentFrame();
  const { durationInFrames } = useVideoConfig();
  const push = interpolate(frame, [0, durationInFrames], [1.0, 1.08]);
  const h = highlight ? m.highlight : null;
  // Fit the highlight box (or the whole image) into the band, scaled to the band's width.
  const box = h ?? { x: 0, y: 0, w: m.width, h: m.height };
  const pad = h ? 0.25 : 0;
  const bw = box.w * (1 + pad), bh = box.h * (1 + pad);
  const scale = Math.min(W / bw, MEDIA.height / bh) * (h ? 1 : 1);
  // Photos fill the band and very wide pages fit, unless the beat asks: fit "contain" keeps the whole image (charts, cards).
  const cover = !h && (fit ? fit === "cover" : m.width / m.height < W / MEDIA.height * 1.6);
  const s = (cover ? Math.max(W / m.width, MEDIA.height / m.height) : scale) * push;
  const cx = box.x + box.w / 2, cy = box.y + box.h / 2;
  return (
    <div style={{ position: "absolute", left: 0, top: MEDIA.top, width: W, height: MEDIA.height, overflow: "hidden",
      borderTop: current.light ? "2px solid rgba(31,32,35,0.15)" : "none", borderBottom: current.light ? "2px solid rgba(31,32,35,0.15)" : "none",
      background: current.light ? "#fff" : "#000" }}>
      <Img src={staticFile(m.src)} style={{ position: "absolute", width: m.width * s, height: m.height * s,
        left: W / 2 - cx * s, top: MEDIA.height / 2 - cy * s }} />
      {h && <div style={{ position: "absolute", left: W / 2 - (box.w * s) / 2 - 10, top: MEDIA.height / 2 - (box.h * s) / 2 - 8,
        width: box.w * s + 20, height: box.h * s + 16, border: `5px solid ${BRAND.yellow}`, borderRadius: 10 }} />}
      {stamp && <div style={{ position: "absolute", right: 24, top: 22, background: "rgba(0,0,0,0.75)", color: BRAND.yellow,
        fontWeight: 800, fontSize: 30, padding: "8px 16px", borderRadius: 8 }}>{stamp}</div>}
      {badge && <div style={{ position: "absolute", left: 24, bottom: 70, background: BRAND.red, color: "#fff", fontWeight: 900,
        fontSize: 54, padding: "10px 22px", borderRadius: 12 }}>{badge}</div>}
      {credit && <div style={{ position: "absolute", left: 24, bottom: 18, fontSize: 24, color: "#fff", background: "rgba(0,0,0,0.6)",
        padding: "4px 12px", borderRadius: 6 }}>{credit}</div>}
    </div>
  );
};

// A few seconds of muted public footage (tools/clip.py) filling the media band, always credited (checklist HARD 7.2).
// Past the end of the footage the last frame holds with a slow push-in.
const ClipBand: React.FC<{ m: Media; credit?: string; stamp?: string }> = ({ m, credit, stamp }) => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const clipF = Math.min(durationInFrames, Math.floor((m.duration ?? durationInFrames / fps) * fps));
  const hold = interpolate(frame, [clipF, Math.max(clipF + 1, durationInFrames)], [1, 1.06], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const media: React.CSSProperties = { position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" };
  return (
    <div style={{ position: "absolute", left: 0, top: MEDIA.top, width: W, height: MEDIA.height, overflow: "hidden", background: "#000" }}>
      <Sequence durationInFrames={clipF} layout="none">
        <OffthreadVideo src={staticFile(m.src)} muted style={media} />
      </Sequence>
      {frame >= clipF && m.last && <Img src={staticFile(m.last)} style={{ ...media, transform: `scale(${hold})` }} />}
      {stamp && <div style={{ position: "absolute", right: 24, top: 22, background: "rgba(0,0,0,0.75)", color: BRAND.yellow,
        fontWeight: 800, fontSize: 30, padding: "8px 16px", borderRadius: 8 }}>{stamp}</div>}
      <div style={{ position: "absolute", left: 24, bottom: 18, fontSize: 24, color: "#fff", background: "rgba(0,0,0,0.6)",
        padding: "4px 12px", borderRadius: 6 }}>{credit ?? m.credit}</div>
    </div>
  );
};

const Stack: React.FC<{ items: { src: Media; label?: string }[]; credit?: string }> = ({ items, credit }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const n = Math.min(items.length, 2), each = (MEDIA.height - 16) / n;
  return (
    <div style={{ position: "absolute", left: 0, top: MEDIA.top, width: W, height: MEDIA.height }}>
      {items.slice(0, n).map((it, i) => {
        const inn = spring({ frame: frame - i * 10, fps, config: { damping: 16 } });
        const s = Math.max(W / it.src.width, each / it.src.height);
        return (
          <div key={i} style={{ position: "absolute", left: 0, top: i * (each + 16), width: W, height: each, overflow: "hidden", opacity: inn }}>
            <Img src={staticFile(it.src.src)} style={{ position: "absolute", width: it.src.width * s, height: it.src.height * s,
              left: (W - it.src.width * s) / 2, top: (each - it.src.height * s) / 2 }} />
            {it.label && <div style={{ position: "absolute", left: 24, top: 20, background: "rgba(0,0,0,0.75)", color: "#fff", fontWeight: 800,
              fontSize: 34, padding: "8px 16px", borderRadius: 8 }}>{it.label}</div>}
          </div>
        );
      })}
      {credit && <div style={{ position: "absolute", left: 24, bottom: 18, fontSize: 24, color: "#fff", background: "rgba(0,0,0,0.6)",
        padding: "4px 12px", borderRadius: 6 }}>{credit}</div>}
    </div>
  );
};

// Text cards fill the media band, centred, so a text beat never leaves a big empty stage (QA of the first Shorts).
const Band: React.FC<{ children: React.ReactNode; style?: React.CSSProperties }> = ({ children, style }) => (
  <div style={{ position: "absolute", left: SAFE.left, right: SAFE.right, top: MEDIA.top, height: MEDIA.height, display: "flex",
    flexDirection: "column", justifyContent: "center", ...style }}>{children}</div>
);

const Kinetic: React.FC<{ text: string; sub?: string }> = ({ text, sub }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const inn = spring({ frame, fps, config: { damping: 14 } });
  const size = text.length <= 18 ? 132 : text.length <= 32 ? 112 : text.length <= 48 ? 96 : 82;
  return (
    <Band style={{ transform: `scale(${0.92 + 0.08 * inn})`, opacity: inn }}>
      <div style={{ width: 120, height: 14, borderRadius: 7, background: current.light ? BRAND.red : BRAND.yellow, marginBottom: 40 }} />
      <div style={{ fontSize: size, fontWeight: 900, lineHeight: 1.02, letterSpacing: -2 }}>{text}</div>
      {sub && <div style={{ fontSize: 52, marginTop: 34, color: C.dim, fontWeight: 600, lineHeight: 1.2 }}>{sub}</div>}
    </Band>
  );
};

// Thousands separators, fixed decimals: 29990 -> "29,990".
const num = (v: number, d = 0) => v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });

// Shrink a long number so it stays inside the safe width (heavy digits run ~0.62 em wide).
const statSize = (text: string, base: number) => Math.min(base, Math.floor((W - SAFE.left - SAFE.right) / (text.length * 0.62)));

// Numbers appear at their final value with a pop: a count-up shows wrong in-between values (e.g. "$0.00M"), which
// a viewer can screenshot or misread.
const Stat: React.FC<{ items: { to: number; prefix?: string; suffix?: string; label: string; decimals?: number }[] }> = ({ items }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const two = items.length > 1;
  return (
    <Band style={{ gap: two ? 90 : 0 }}>
      {items.slice(0, 2).map((it, i) => {
        const p = spring({ frame: frame - i * 12, fps, config: { damping: 12, mass: 0.6 } });
        return (
          <div key={i} style={{ opacity: Math.min(1, p * 1.5), transform: `scale(${0.85 + 0.15 * p})`, transformOrigin: "left center" }}>
            <div style={{ fontSize: statSize(`${it.prefix ?? ""}${num(it.to, it.decimals)}${it.suffix ?? ""}`, two ? 190 : 250), fontWeight: 900, letterSpacing: -5, color: current.light ? BRAND.red : BRAND.yellow, lineHeight: 1 }}>
              {it.prefix ?? ""}{num(it.to, it.decimals)}{it.suffix ?? ""}
            </div>
            <div style={{ fontSize: 52, fontWeight: 700, marginTop: 14, lineHeight: 1.2 }}>{it.label}</div>
          </div>
        );
      })}
    </Band>
  );
};

// Captions: 2-4 words at a time, big, in the clear band between the picture and the Shorts UI.
const Captions: React.FC<{ words: Word[] }> = ({ words }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const now = frame / fps;
  const chunks: number[][] = [];
  let cur: number[] = [];
  words.forEach((w, i) => {
    cur.push(i);
    if (cur.length >= 4 || (/[.,?!:;]$/.test(w.text) && cur.length >= 2)) { chunks.push(cur); cur = []; }
  });
  if (cur.length) chunks.push(cur);
  let active = -1;
  for (let i = 0; i < words.length; i++) if (now >= words[i].start) active = i;
  if (active < 0 || now > words[words.length - 1].end + 0.3) return null;
  const chunk = chunks.find((c) => c.includes(active))!;
  return (
    <div style={{ position: "absolute", left: SAFE.left, right: SAFE.right, top: MEDIA.top + MEDIA.height + 22, display: "flex", justifyContent: "center" }}>
      <div style={{ background: "rgba(0,0,0,0.82)", borderRadius: 16, padding: "14px 26px", fontSize: 62, fontWeight: 800,
        letterSpacing: -0.5, textAlign: "center", lineHeight: 1.15 }}>
        {/* Real spaces between words (not margins), so a long group can wrap inside the safe area. */}
        {chunk.map((i, k) => (
          <React.Fragment key={i}>
            {k > 0 && " "}
            <span style={{ color: i === active ? BRAND.yellow : "#f5f4f2" }}>{words[i].text}</span>
          </React.Fragment>
        ))}
      </div>
    </div>
  );
};

const EndCard: React.FC<{ cta: string }> = ({ cta }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const inn = spring({ frame, fps, config: { damping: 14 } });
  return (
    <AbsoluteFill style={{ alignItems: "center", justifyContent: "center", opacity: inn }}>
      <div style={{ position: "absolute", top: 560, left: SAFE.left, right: SAFE.right, display: "flex", flexDirection: "column", alignItems: "center", gap: 40 }}>
        <Lockup size={84} />
        <div style={{ fontSize: 58, fontWeight: 800, textAlign: "center", lineHeight: 1.15 }}>{cta}</div>
      </div>
    </AbsoluteFill>
  );
};
