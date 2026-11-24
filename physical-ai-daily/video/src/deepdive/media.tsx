import React from "react";
import { Freeze, Img, OffthreadVideo, Sequence, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { C, Media, STAGE } from "../theme";
import { BRAND } from "../brand";
import { current } from "../design";
import { Tag } from "./chrome";
import { Cue, DP, Icon, SETTLED, clauseAt, cuer, ease, firstIn, hi, line, newest, onHi, pop, revealAt, settle } from "./kit";

// Footage: video (one clip, punched in, aligned, frozen, marked), pair (sim vs real side by side, or in sequence), grid
// (2x2 / 2x3 tiles). Clip time is remapped per side: start offset ("from"), speed, a freeze, a hold on the last frame.
// Clips are always muted. Emphasis is a border and label colour; a clip being discussed is never dimmed.

type Crop = { x: number; y: number; w: number; h: number };
type Mark = { side?: number; x: number; y: number; shape?: "ring" | "arrow"; label?: string };
type Travel = "left" | "right" | "up" | "down" | { dir: "left" | "right" | "up" | "down"; x: number; y: number };
export type Side = Media & {
  label?: string; outcome?: string; ok?: boolean; outcomeCue?: Cue; stamp?: string; cue?: Cue; reveal?: Cue;
  from?: number; to?: number; playbackRate?: number; crop?: Crop; travel?: Travel; check?: { cue?: Cue; ok?: boolean; x?: number; y?: number };
};
type FreezeSpec = { cue?: Cue; s?: number; at?: number | number[]; marks?: Mark[] };

const isClip = (src: string) => /\.(mp4|webm|mov|m4v)$/i.test(src);
const SHADOW = () => (current.light ? "0 18px 50px rgba(31,32,35,0.18), 0 0 0 1px rgba(31,32,35,0.08)" : "0 30px 80px rgba(0,0,0,0.55)");
const LABEL_ROW = 66;

// Clip seconds shown at a scene frame: from + speed x time since the clip started (t0), held during a freeze (at a
// given clip time, or wherever it was), clamped to "to" / the clip's end (where its last frame holds).
type Clock = { t: number; natural: boolean };
const clockOf = (it: Side, t0: number, fz: { f0: number; n: number; at?: number } | null, frame: number, fps: number): Clock => {
  const rate = it.playbackRate ?? 1, from = it.from ?? 0, len = it.duration ?? 1e9;
  const end = Math.min(it.to ?? len, len - 0.04);
  const run = (f: number) => from + (rate * Math.max(0, f - t0)) / fps;
  let t = run(frame);
  if (fz && frame >= fz.f0) {
    const held = fz.at ?? run(fz.f0);
    t = frame < fz.f0 + fz.n ? held : held + (rate * (frame - fz.f0 - fz.n)) / fps;
  }
  return { t: Math.min(Math.max(0, t), end), natural: it.to == null && t >= end };
};

// A crop (clip pixels) punched in to fill a w x h frame: the whole crop shows, the clip around it fills the rest.
const geo = (it: Side, w: number, h: number) => {
  const cr = it.crop ?? { x: 0, y: 0, w: it.width, h: it.height };
  const s = Math.min(w / cr.w, h / cr.h), W = it.width * s, H = it.height * s;
  const fitAxis = (size: number, full: number, centre: number) =>
    full >= size ? Math.min(0, Math.max(size - full, size / 2 - centre * s)) : (size - full) / 2;
  const left = fitAxis(w, W, cr.x + cr.w / 2), top = fitAxis(h, H, cr.y + cr.h / 2);
  return { s, W, H, left, top, at: (x: number, y: number) => ({ x: left + x * s, y: top + y * s }) };
};
const aspect = (it: Side) => (it.crop ? it.crop.h / it.crop.w : it.height / it.width);

const MediaLayer: React.FC<{ it: Side; w: number; h: number; clock: Clock; fps: number; fit?: "contain" | "cover" }> = ({ it, w, h, clock, fps, fit }) => {
  const g = geo(it, w, h);
  const box: React.CSSProperties = fit === "cover" && !it.crop
    ? { position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "cover" }
    : { position: "absolute", left: g.left, top: g.top, width: g.W, height: g.H };
  if (!isClip(it.src)) return <Img src={staticFile(it.src)} style={box} />;
  if (clock.natural && it.last) return <Img src={staticFile(it.last)} style={box} />;
  return (
    <Freeze frame={Math.round(clock.t * fps * 1000) / 1000}>
      <Sequence from={0} layout="none"><OffthreadVideo src={staticFile(it.src)} muted style={box} /></Sequence>
    </Freeze>
  );
};

// A red marker landing on the divergence point: a ring around it, or an arrow at it.
const Marker: React.FC<{ m: Mark; at: { x: number; y: number }; p: number }> = ({ m, at, p }) => {
  if (p <= 0) return null;
  const red = BRAND.red;
  return (
    <>
      <svg style={{ position: "absolute", left: 0, top: 0, overflow: "visible", opacity: Math.min(1, p) }} width={1} height={1}>
        {m.shape === "arrow" ? (
          <g transform={`translate(${at.x} ${at.y}) scale(${0.7 + 0.3 * p})`}>
            <line x1={-120} y1={-120} x2={-22} y2={-22} stroke="#fff" strokeWidth={15} strokeLinecap="round" />
            <line x1={-120} y1={-120} x2={-22} y2={-22} stroke={red} strokeWidth={8} strokeLinecap="round" />
            <polygon points="-4,-4 -48,-18 -18,-48" fill={red} stroke="#fff" strokeWidth={3} />
          </g>
        ) : (
          <g transform={`translate(${at.x} ${at.y}) scale(${0.7 + 0.3 * p})`}>
            <circle r={50} fill="none" stroke="#fff" strokeWidth={14} />
            <circle r={50} fill="none" stroke={red} strokeWidth={8} />
          </g>
        )}
      </svg>
      {m.label && (
        <div style={{ position: "absolute", left: at.x + 58, top: at.y - 22, padding: "4px 14px", borderRadius: 8, background: red, color: "#fff",
          fontSize: 30, fontWeight: 800, whiteSpace: "nowrap", opacity: Math.min(1, p) }}>{m.label}</div>
      )}
    </>
  );
};

// Direction of travel: a white arrow with a dark edge, at the bottom centre of the frame or at x, y (clip pixels).
const ROT = { right: 0, down: 90, left: 180, up: 270 };
const TravelArrow: React.FC<{ tr: Travel; at: { x: number; y: number } }> = ({ tr, at }) => {
  const dir = typeof tr === "string" ? tr : tr.dir;
  return (
    <svg style={{ position: "absolute", left: 0, top: 0, overflow: "visible" }} width={1} height={1}>
      <g transform={`translate(${at.x} ${at.y}) rotate(${ROT[dir]})`} opacity={0.92}>
        <path d="M-75 -9 H35 V-26 L78 0 L35 26 V9 H-75 Z" fill="#fff" stroke="rgba(0,0,0,0.6)" strokeWidth={3} strokeLinejoin="round" />
      </g>
    </svg>
  );
};

// One side: the framed clip with its crop, marks, goal check and travel arrow; its label (and outcome) above the frame.
const Panel: React.FC<{ it: Side; x: number; y: number; w: number; h: number; clock: Clock; on: boolean; fps: number; frame: number;
  marks: Mark[]; markP: number; checkAt: number; outcomeAt: number; o?: number }> = ({ it, x, y, w, h, clock, on, fps, frame, marks, markP, checkAt, outcomeAt, o = 1 }) => {
  const g = geo(it, w, h);
  const ck = it.check && frame >= checkAt ? pop(frame, checkAt, fps) : 0; // (an old clip holding its slot has none)
  const ckAt = it.check?.x != null ? g.at(it.check.x, it.check.y ?? 0) : { x: w - 60, y: 60 };
  const tr = it.travel ? (typeof it.travel === "string" ? { x: w / 2, y: h - 52 } : g.at(it.travel.x, it.travel.y)) : null;
  const oc = frame >= outcomeAt ? ease(frame, outcomeAt, 8) : 0;
  return (
    <div style={{ opacity: o }}>
      {it.label && (
        <div style={{ position: "absolute", left: x - 60, width: w + 120, top: y - 62, textAlign: "center", whiteSpace: "nowrap" }}>
          <span style={{ fontSize: 38, fontWeight: 800, letterSpacing: -0.4, color: on ? hi() : C.ink, verticalAlign: "middle" }}>{it.label}</span>
          {it.outcome && (
            <span style={{ display: "inline-flex", alignItems: "center", gap: 8, marginLeft: 16, padding: "2px 16px 2px 8px", borderRadius: 22,
              fontSize: 32, fontWeight: 700, color: C.ink, border: `2px solid ${line()}`, background: current.light ? "#fff" : C.panel2,
              opacity: oc, verticalAlign: "middle" }}>
              {it.ok != null && <Icon name={it.ok ? "check" : "cross"} size={32} color={it.ok ? C.ink : BRAND.red} />}{it.outcome}
            </span>
          )}
        </div>
      )}
      <div style={{ position: "absolute", left: x, top: y, width: w, height: h, borderRadius: 14, overflow: "hidden", background: "#000",
        boxShadow: SHADOW(), outline: on ? `5px solid ${hi()}` : "none", outlineOffset: 4 }}>
        <MediaLayer it={it} w={w} h={h} clock={clock} fps={fps} />
        {tr && <TravelArrow tr={it.travel!} at={tr} />}
        {marks.map((m, i) => <Marker key={i} m={m} at={g.at(m.x, m.y)} p={markP} />)}
        {ck > 0 && (
          <div style={{ position: "absolute", left: ckAt.x - 40, top: ckAt.y - 40, width: 80, height: 80, borderRadius: 40, display: "flex",
            alignItems: "center", justifyContent: "center", background: it.check!.ok === false ? BRAND.red : "#16a34a",
            boxShadow: "0 0 0 5px rgba(255,255,255,0.9)", transform: `scale(${ck})` }}>
            <Icon name={it.check!.ok === false ? "cross" : "check"} size={54} color="#fff" />
          </div>
        )}
        {it.stamp && <Tag text={it.stamp} at={{ right: 14, top: h - 54 }} />}
      </div>
    </div>
  );
};

// The freeze of a pair or video: its start and length in frames, and the clip time each side holds.
const freezeOf = (p: DP, fps: number, fz: FreezeSpec | undefined) => {
  if (!fz) return null;
  const f0 = cuer(p, fps)(fz.cue, p.pre + Math.round(p.len * 0.6));
  return { f0, n: Math.round((fz.s ?? 1.5) * fps), at: (side: number) => (Array.isArray(fz.at) ? fz.at[side] : side === 0 ? fz.at : undefined) };
};
const markP = (frame: number, fps: number, f: ReturnType<typeof freezeOf>) =>
  !f || frame < f.f0 ? 0 : frame < f.f0 + f.n ? pop(frame, f.f0, fps) : Math.max(0, 1 - (frame - f.f0 - f.n) / 6);

// Which tile the narration is on: the latest focus step, moved to the start of the clause that names it.
const focusOf = (p: DP, fps: number, frame: number) => {
  const focus: { item: number; cue?: Cue }[] = p.v.focus ?? [];
  const fAt = focus.map((f) => clauseAt(p, fps, f.cue, p.pre + Math.round(p.len * 0.6)));
  const fk = newest(fAt, frame);
  return fk >= 0 ? focus[fk].item : -1;
};

// A frame of the given aspect, as large as fits w x h, centred there.
const contain = (ratio: number, x: number, y: number, w: number, h: number) => {
  const fw = Math.min(w, h / ratio), fh = fw * ratio;
  return { x: x + (w - fw) / 2, y: y + (h - fh) / 2, w: fw, h: fh };
};

// ---------- video ----------

// One clip, as a deep-dive shot: crop (punch in), from / to / playbackRate, freeze {cue, s, at, marks}, check, travel,
// label / outcome, stamp. The clip starts with its beat (from its lead-in it shows its first frame).
export const Video: React.FC<DP> = (p) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const v = p.v, c = cuer(p, fps);
  const it: Side = { ...v.src, ...v, src: v.src.src };
  const fz = freezeOf(p, fps, v.freeze);
  const clock = clockOf(it, p.pre, fz ? { f0: fz.f0, n: fz.n, at: fz.at(0) } : null, frame, fps);
  const b = contain(aspect(it), STAGE.x, STAGE.y + (it.label ? LABEL_ROW : 0), STAGE.w, STAGE.h - (it.label ? LABEL_ROW : 0));
  return <Panel it={it} {...b} clock={clock} on={false} fps={fps} frame={frame} marks={v.freeze?.marks ?? []} markP={markP(frame, fps, fz)}
    checkAt={c(it.check?.cue, p.pre + Math.round(p.len * 0.8))} outcomeAt={settle(p, c(it.outcomeCue, 0))} />;
};

// ---------- swapping in place ----------

// What a pair side or grid tile shows at a frame. A beat with the same layout as the one before (DeepDive passes it as
// prev) swaps clips in place: a slot keeps the old clip until its new one's word, and a clip that stays in its slot keeps
// playing (from where it was, unless the new beat sets "from").
type SlotView = { it: Side; clock: Clock } | null;
const pairAt = (p: DP, fps: number) => {
  const v = p.v, c = cuer(p, fps), seq = v.mode === "sequence";
  return (v.items ?? []).slice(0, 2).map((it: Side, i: number) => seq ? (i === 0 ? SETTLED : c(it.cue, p.pre + Math.round(p.len * 0.35)))
    : it.reveal != null ? settle(p, c(it.reveal, p.pre)) : SETTLED);
};
const gridAt = (p: DP, fps: number) => {
  const at = revealAt((p.v.items ?? []).slice(0, 6), p, fps).map((a) => settle(p, a));
  return p.prev ? at : firstIn(at); // the first tile is in place at the cut; on a swap, the old tiles hold the places
};
const slotView = (p: DP, i: number, frame: number, fps: number): SlotView => {
  const v = p.v, grid = v.type === "grid", items: Side[] = v.items ?? [], it = items[i];
  if (!it) return null;
  const at = grid ? gridAt(p, fps) : pairAt(p, fps);
  const old: Side | undefined = p.prev?.v.items?.[i];
  const same = !!old && old.src === it.src;
  if (p.prev && old && !same && frame < at[i]) return slotView(p.prev, i, p.prev.dur + frame, fps);
  if (frame < at[i] && !same) return null;
  // a clip staying in its slot continues: its clip time where the last beat left it
  const cont = same && it.from == null ? { ...it, from: slotView(p.prev!, i, p.prev!.dur, fps)?.clock.t ?? 0 } : it;
  const seqA = !grid && v.mode === "sequence" && i === 0 && it.cue != null;
  const t0 = Math.max(p.pre, same ? 0 : seqA ? cuer(p, fps)(it.cue, p.pre) : at[i]);
  const fz = grid ? null : freezeOf(p, fps, v.freeze);
  return { it, clock: clockOf(cont, t0, fz ? { f0: fz.f0, n: fz.n, at: fz.at(i) } : null, frame, fps) };
};

// ---------- pair ----------

// Sim vs real. Split (default): two equal frames side by side, both on screen and at full opacity throughout (a "reveal"
// cue holds one back); focus lights the side the narration is on with a border and label colour. Per side: crop, from,
// playbackRate, outcome, check, travel, stamp. freeze: on its cue both sides hold (at freeze.at[side] clip seconds, or
// where they are) for s seconds and the marks land on the divergence. mode "sequence": side A full frame, then side B
// full frame on items[1].cue, then the split on "split" (usually with the freeze), each a cut. Consecutive split pairs
// swap clips in place.
export const Pair: React.FC<DP> = (p) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const v = p.v, c = cuer(p, fps);
  const items: Side[] = (v.items ?? []).slice(0, 2);
  const seq = v.mode === "sequence";
  const at = pairAt(p, fps);
  const splitAt = seq ? c(v.split, p.pre + Math.round(p.len * 0.7)) : -Infinity;
  const fz = freezeOf(p, fps, v.freeze);
  const lit = focusOf(p, fps, frame);
  const mp = markP(frame, fps, fz);
  const views = items.map((_, i) => slotView(p, i, frame, fps));
  const ratio = Math.max(...items.map(aspect));
  const gap = 48, half = (STAGE.w - gap) / 2, top = STAGE.y + LABEL_ROW, H = STAGE.h - LABEL_ROW;
  const phase = !seq || frame >= splitAt ? "split" : frame >= at[1] ? 1 : 0;
  const panel = (i: number, b: { x: number; y: number; w: number; h: number }) => {
    const sv = views[i];
    if (!sv) return null;
    const fresh = sv.it === items[i]; // (on a swap, the old clip holds the slot until this one's word)
    return (
      <Panel key={i} it={sv.it} {...b} clock={sv.clock} on={i === lit} fps={fps} frame={frame}
        marks={fresh ? (v.freeze?.marks ?? []).filter((m: Mark) => (m.side ?? 0) === i) : []} markP={mp}
        checkAt={fresh ? c(items[i].check?.cue, p.pre + Math.round(p.len * 0.8)) : Infinity} outcomeAt={settle(p, c(sv.it.outcomeCue, 0))}
        o={p.prev || seq || at[i] === SETTLED ? 1 : Math.min(1, ease(frame, at[i], 8))} />
    );
  };
  if (phase !== "split") return panel(phase, contain(aspect(items[phase]), STAGE.x, top, STAGE.w, H));
  return <>{items.map((_, i) => panel(i, contain(ratio, STAGE.x + i * (half + gap), top, half, H)))}</>;
};

// ---------- grid ----------

// 2x2 or 2x3 clips or images, each labelled on its tile (optional stamp, crop, from, playbackRate). The first tile is in
// place at the cut and the others land on their cues (no empty frames before); one focus at a time is a border and label
// colour. Consecutive grids with the same layout swap clips in place, each slot on its new clip's word.
export const Grid: React.FC<DP> = (p) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const v = p.v;
  const items: Side[] = (v.items ?? []).slice(0, 6);
  const n = Math.max(1, items.length), cols = v.cols ?? (n <= 4 ? 2 : 3), rows = Math.ceil(n / cols);
  const at = gridAt(p, fps);
  const lit = focusOf(p, fps, frame);
  const top = 152, H = 856 - top, gap = 20; // under the chapter header, above the credit chip
  const tw = Math.min((STAGE.w - gap * (cols - 1)) / cols, (((H - gap * (rows - 1)) / rows) * 16) / 9), th = (tw * 9) / 16;
  const gw = cols * tw + (cols - 1) * gap, gh = rows * th + (rows - 1) * gap;
  const x0 = 960 - gw / 2, y0 = top + (H - gh) / 2;
  return (
    <>
      {items.map((_, i) => {
        const sv = slotView(p, i, frame, fps);
        if (!sv) return null;
        const x = x0 + (i % cols) * (tw + gap), y = y0 + Math.floor(i / cols) * (th + gap);
        const s = sv.it !== items[i] || frame >= at[i] + 30 || p.prev ? 1 : pop(frame, at[i], fps); // a swap cuts, a new tile pops in
        const on = i === lit;
        return (
          <div key={i} style={{ position: "absolute", left: x, top: y, width: tw, height: th, borderRadius: 12, overflow: "hidden", background: "#000",
            boxShadow: SHADOW(), outline: on ? `5px solid ${hi()}` : "none", outlineOffset: 3, opacity: Math.min(1, s),
            transform: `scale(${0.96 + 0.04 * Math.min(1, s)})` }}>
            <MediaLayer it={sv.it} w={tw} h={th} clock={sv.clock} fps={fps} fit={v.fit} />
            <div style={{ position: "absolute", left: 12, top: 12, padding: "5px 14px", borderRadius: 8, fontSize: 30, fontWeight: 800,
              background: on ? hi() : "rgba(15,15,17,0.72)", color: on ? onHi() : "#fff", maxWidth: tw - 24, whiteSpace: "nowrap", overflow: "hidden",
              textOverflow: "ellipsis" }}>{sv.it.label}</div>
            {sv.it.stamp && <Tag text={sv.it.stamp} at={{ right: 12, top: th - 58 }} />}
          </div>
        );
      })}
    </>
  );
};
