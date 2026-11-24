import React from "react";
import { Easing, interpolate, spring } from "remotion";
import { C, STAGE, Word } from "../theme";
import { BRAND } from "../brand";
import { current } from "../design";
import { cueFrame } from "../scenes";
import type { DDSegment } from "./DeepDive";

// Shared pieces of the deep-dive scenes: cue timing, the one highlight colour, box styles, line icons, path geometry.

export type Cue = string | number;
// Every deep-dive scene gets the digest's scene props, the chapter it plays in, and its place on the cut timeline.
// A scene starts `pre` frames before its beat (the crossfade lead) and its words are shifted to match, so every frame
// number below is a frame of the scene's own Sequence.
export type DP = {
  v: any; dur: number; index: number; words: Word[]; seg: DDSegment; chapters: number;
  pre: number; // frames before the beat that this scene is already on screen
  len: number; // the beat's own frames (narration), without the lead or the tail
  hold: number; // last frame a reveal may land on: the beat's end - 1 s, so every finished build is seen whole
  born: number; // entrance animations start here: 0, or SETTLED on the video's first frame (already in place)
  chain?: any[]; // build: the visuals of a cont chain, which share one fit
  prev?: DP; // cont: the beat this one continues (its last highlight carries over)
};

export const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;
export const MONO = "'SF Mono', Menlo, monospace";

// Frame of a cue in this beat: a word (as ../scenes cueFrame: the first word starting with it), "word#2" for its second
// occurrence, or a number of seconds into the beat.
export const cueAt = (words: Word[], cue: Cue | undefined, fps: number, fallback: number) => {
  if (typeof cue === "number") return Math.round(cue * fps);
  const m = cue?.match(/^(.+)#(\d+)$/);
  if (!m) return cueFrame(words, cue, fps, fallback);
  const n = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, "");
  const w = words.filter((w) => n(w.text).startsWith(n(m[1])))[+m[2] - 1];
  return w ? Math.round(w.start * fps) : fallback;
};

// A scene's cue resolver: word (or seconds into the beat) -> scene frame, never later than the hold (beat end - 1 s).
export const cuer = (p: DP, fps: number) => (cue: Cue | undefined, fallback: number) =>
  Math.min(p.hold, typeof cue === "number" ? Math.round(cue * fps) + p.pre : cueAt(p.words, cue, fps, fallback));

// Reveal frames for a list: each item's cue, else spread over the first 60% of the beat. Each part appears on its word,
// never before (diagrams show a faint scaffold of what's to come, so the stage isn't empty meanwhile).
export const revealAt = (items: { cue?: Cue }[], p: DP, fps: number) => {
  const c = cuer(p, fps);
  return items.map((it, i) => c(it.cue, p.pre + Math.round((i / Math.max(1, items.length)) * p.len * 0.6)));
};

// Parts cued at 0 are already in place (no pop) when the scene continues the previous beat's diagram ("cont") or is the
// video's first frame. Settled parts don't take the highlight: the beat's first new reveal (or focus) does.
export const SETTLED = -60;
export const settle = (p: DP, at: number) => ((p.v.cont || p.born < 0) && at <= 0 ? SETTLED : at);

// Emphasis follows the narration: a focus moves at the start of the clause that names its subject ("and for real"
// switches on "and", not on "real"), when that start is within 1 s before the cue word.
export const clauseAt = (p: DP, fps: number, cue: Cue | undefined, fallback: number) => {
  const f = cuer(p, fps)(cue, fallback);
  if (typeof cue !== "string") return f;
  const n = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, "");
  const m = cue.match(/^(.+)#(\d+)$/), stem = n(m ? m[1] : cue), nth = m ? +m[2] : 1;
  const i = p.words.map((w, k) => (n(w.text).startsWith(stem) ? k : -1)).filter((k) => k >= 0)[nth - 1];
  if (i == null) return f;
  let j = i;
  while (j > 0 && !/[.,;:?!]$/.test(p.words[j - 1].text) && p.words[i].start - p.words[j - 1].start <= 1) j--;
  return Math.min(f, Math.round(p.words[j].start * fps));
};

export const pop = (frame: number, at: number, fps: number) => spring({ frame: frame - at, fps, config: { damping: 15, mass: 0.6 } });
export const calm = (frame: number, at: number, fps: number) => spring({ frame: frame - at, fps, config: { damping: 200 } });
export const ease = (frame: number, at: number, len = 14) =>
  interpolate(frame - at, [0, len], [0, 1], { ...clamp, easing: Easing.out(Easing.cubic) });
// Index of the item revealed most recently by this frame (-1 before the first).
export const newest = (at: number[], frame: number) => at.reduce((k, a, i) => (frame >= a && (k < 0 || a >= at[k]) ? i : k), -1);

// One highlight colour per design, as the mono section chip: brand red on paper, yellow on black.
export const hi = () => (current.light ? BRAND.red : C.accent);
export const onHi = () => (current.light ? "#fff" : "#111");
export const tint = (a: number) => (current.light ? `rgba(244,63,46,${a})` : `rgba(240,201,69,${a})`);
export const line = () => (current.light ? "rgba(31,32,35,0.22)" : "rgba(255,255,255,0.2)");
export const ink2 = () => (current.light ? "rgba(31,32,35,0.62)" : "rgba(245,244,242,0.55)"); // arrows, axes
export const DIM = 0.26; // opacity of what the narration isn't on
export const LABEL_PX = 38, SUB_PX = 28, MAX_TEXT_PX = 64; // diagram text at scale 1; the auto-fit never goes past 64 px
export const dimWash = () => (current.light ? "rgba(251,250,247,0.74)" : "rgba(10,10,11,0.64)");
export const pad2 = (n: number) => String(n).padStart(2, "0");

// A part not yet named: a faint dashed outline where it will land, so a diagram never opens on an empty stage and
// nothing is shown before it's said.
export const ghostStyle = (): React.CSSProperties => ({
  border: `2px dashed ${current.light ? "rgba(31,32,35,0.2)" : "rgba(255,255,255,0.18)"}`, borderRadius: 16, boxSizing: "border-box",
});

// Auto-fit: a diagram's bounding box (stage-local) scaled to fill about 80% of STAGE and centred, never past 64 px text,
// never below scale 1 unless it wouldn't fit the stage at all. One-box diagrams grow instead of sitting small.
export type Fit = { s: number; tx: number; ty: number };
export const fitBox = (x0: number, y0: number, x1: number, y1: number): Fit => {
  const bw = Math.max(1, x1 - x0), bh = Math.max(1, y1 - y0);
  const want = Math.min((STAGE.w * 0.8) / bw, (STAGE.h * 0.8) / bh);
  const s = Math.min(MAX_TEXT_PX / LABEL_PX, want >= 1 ? want : Math.min(1, STAGE.w / bw, STAGE.h / bh));
  return { s, tx: STAGE.w / 2 - (s * (x0 + x1)) / 2, ty: STAGE.h / 2 - (s * (y0 + y1)) / 2 };
};
// The stage, in stage-local coordinates, with a fit applied: diagrams draw inside it.
export const Staged: React.FC<{ fit: Fit; children: React.ReactNode }> = ({ fit, children }) => (
  <div style={{ position: "absolute", left: STAGE.x, top: STAGE.y, width: STAGE.w, height: STAGE.h, transformOrigin: "0 0",
    transform: `translate(${fit.tx}px, ${fit.ty}px) scale(${fit.s})` }}>{children}</div>
);

// A diagram box: white card on paper, raised panel on black; the highlighted one gets the highlight border and a halo.
export const boxStyle = (on: boolean): React.CSSProperties => ({
  background: current.light ? "#fff" : C.panel2, borderRadius: 16, border: `3px solid ${on ? hi() : line()}`,
  boxShadow: (on ? `0 0 0 5px ${tint(0.16)}, ` : "") + (current.light ? "0 6px 20px rgba(31,32,35,0.08)" : "0 10px 30px rgba(0,0,0,0.45)"),
});

// Line icons on a 24 grid (stroke only, so they take the theme colour). Any other string renders as text (an emoji).
const O = (x: number, y: number, r: number) => `M${x - r} ${y}a${r} ${r} 0 1 0 ${2 * r} 0a${r} ${r} 0 1 0 ${-2 * r} 0`;
const ICONS: Record<string, string> = {
  agent: "M3 5h18v14H3z M7 10l3 2-3 2 M12 15h5", // a terminal: coding agent
  code: "M8 7l-5 5 5 5 M16 7l5 5-5 5",
  sim: "M12 3l8 4.5v9L12 21l-8-4.5v-9z M4 7.5l8 4.5 8-4.5 M12 12v9",
  robot: `M6 9h12v10H6z M12 9V6 ${O(12, 4.5, 1.5)} M9.5 13.5v1 M14.5 13.5v1 M3 13v3 M21 13v3`,
  video: "M3 7h12v10H3z M15 10.5L21 7v10l-6-3.5z",
  build: "M14.5 6.5a4 4 0 0 0-5 5L3.5 17.5l3 3 6-6a4 4 0 0 0 5-5l-2.5 2.5-2.5-.5-.5-2.5z",
  test: "M5 21V4 M5 4h12l-2.5 4L17 12H5",
  check: "M5 12.5l4.5 4.5L19 7.5",
  cross: "M6 6l12 12 M18 6L6 18",
  chart: "M4 20V11 M10 20V5 M16 20v-6 M3 20h18",
  data: "M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3z M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6 M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3",
  eye: `M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z ${O(12, 12, 3)}`,
  doc: "M6 3h8l4 4v14H6z M14 3v4h4 M9 12h6 M9 16h6",
  model: `${O(5, 12, 2)} ${O(19, 6, 2)} ${O(19, 18, 2)} M7 11.2l10-4.4 M7 12.8l10 4.4`,
  person: `${O(12, 8, 4)} M4 21c0-4 3.6-7 8-7s8 3 8 7`,
  measure: "M3 8h18v8H3z M7 8v3 M11 8v4 M15 8v3 M19 8v4",
  loop: "M20 12a8 8 0 1 1-2.34-5.66 M20 4v4h-4",
  parts: "M4 8h16v12H4z M4 8l2-4h12l2 4 M10 12h4",
  motor: "M13 2L4 14h7l-1 8 9-12h-7z",
  wall: "M3 20h18 M6 20V9h12v11 M6 14.5h12 M12 9v5.5 M9 14.5V20 M15 14.5V20",
};
export const Icon: React.FC<{ name: string; size: number; color: string }> = ({ name, size, color }) =>
  ICONS[name] ? (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth={1.7} strokeLinecap="round"
      strokeLinejoin="round" style={{ flexShrink: 0, display: "block" }}><path d={ICONS[name]} /></svg>
  ) : <span style={{ fontSize: size * 0.82, lineHeight: 1, flexShrink: 0 }}>{name}</span>;

// ---------- geometry: paths as sampled polylines, so arrows draw on and markers travel at constant speed ----------

export type Pt = { x: number; y: number };
export type Poly = { pts: Pt[]; cum: number[]; len: number; d: string };
export type Box = { x: number; y: number; w: number; h: number }; // centre and size

export const poly = (pts: Pt[]): Poly => {
  const cum = [0];
  for (let i = 1; i < pts.length; i++) cum.push(cum[i - 1] + Math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y));
  return { pts, cum, len: cum[cum.length - 1], d: "M" + pts.map((p) => `${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(" L") };
};
export const sample = (f: (t: number) => Pt, n = 48) => Array.from({ length: n + 1 }, (_, i) => f(i / n));
export const bez = (a: Pt, b: Pt, c: Pt, d: Pt) => (t: number): Pt => {
  const u = 1 - t;
  return { x: u * u * u * a.x + 3 * u * u * t * b.x + 3 * u * t * t * c.x + t * t * t * d.x,
    y: u * u * u * a.y + 3 * u * u * t * b.y + 3 * u * t * t * c.y + t * t * t * d.y };
};
// Point and heading at arc length s.
export const along = (pl: Poly, s: number) => {
  s = Math.max(0, Math.min(pl.len, s));
  let i = 1;
  while (i < pl.cum.length - 1 && pl.cum[i] < s) i++;
  const a = pl.pts[i - 1], b = pl.pts[i] ?? a, seg = pl.cum[i] - pl.cum[i - 1] || 1, t = (s - pl.cum[i - 1]) / seg;
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t, angle: Math.atan2(b.y - a.y, b.x - a.x) };
};
// Where the ray from a box's centre towards (tx, ty) leaves the box, plus a gap.
export const exit = (r: Box, tx: number, ty: number, gap = 10): Pt => {
  const dx = tx - r.x, dy = ty - r.y;
  const k = 1 / Math.max(Math.abs(dx) / (r.w / 2 + gap), Math.abs(dy) / (r.h / 2 + gap), 1e-6);
  return { x: r.x + dx * k, y: r.y + dy * k };
};
export const inside = (r: Box, p: Pt, gap = 10) => Math.abs(p.x - r.x) < r.w / 2 + gap && Math.abs(p.y - r.y) < r.h / 2 + gap;

// An arrow drawing on along a path (p 0..1): the line grows from its start and the head rides the tip.
export const Arrow: React.FC<{ pl: Poly; p: number; color: string; width?: number; dashed?: boolean; id: string }> = ({ pl, p, color, width = 4, dashed, id }) => {
  if (p <= 0 || pl.len < 1) return null;
  const tip = along(pl, pl.len * p);
  const L = 20, W = 11, c = Math.cos(tip.angle), s = Math.sin(tip.angle);
  const head = [[0, 0], [-L, -W], [-L, W]].map(([x, y]) => `${tip.x + x * c - y * s},${tip.y + x * s + y * c}`).join(" ");
  const drawn = Math.max(0, pl.len * p - L * 0.6); // stop under the head, so the line's end doesn't poke through the tip
  const dash = { strokeDasharray: `${pl.len} ${pl.len}`, strokeDashoffset: pl.len - drawn };
  return (
    <g>
      {dashed ? (
        <>
          <mask id={id} maskUnits="userSpaceOnUse" x={-4000} y={-4000} width={8000} height={8000}>
            <path d={pl.d} stroke="#fff" strokeWidth={width + 10} fill="none" {...dash} />
          </mask>
          <path d={pl.d} stroke={color} strokeWidth={width} fill="none" strokeDasharray="13 10" mask={`url(#${id})`} />
        </>
      ) : <path d={pl.d} stroke={color} strokeWidth={width} fill="none" strokeLinejoin="round" {...dash} />}
      <polygon points={head} fill={color} />
    </g>
  );
};
