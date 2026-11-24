import React from "react";
import { interpolate, useCurrentFrame, useVideoConfig } from "remotion";
import { C, STAGE } from "../theme";
import { current } from "../design";
import {
  Arrow, Box, Cue, DIM, DP, Fit, Icon, LABEL_PX, MONO, Poly, Pt, SETTLED, SUB_PX, Staged, along, bez, boxStyle, clamp, cuer, ease,
  exit, firstIn, fitBox, ghostStyle, hi, inside, ink2, line, newest, onHi, poly, pop, revealAt, sample, settle,
} from "./kit";

// Our own diagrams, built step by step with the narration: build (generic boxes and arrows), loop / steps (an
// iterative process), timeline (a lineage), bars (a bar chart). build / loop / steps are laid out in stage-local
// coordinates (0-1680 x 0-660) and then auto-fitted to the stage.

const GAP = 10; // between a box and the arrow that meets it
const EDGE_PX = 28;

// Box height for its text (38 px label, 28 px sub; bold glyphs about 0.55 em wide), so nothing spills out of a box.
// tools/deepdive/dd.py mirrors it for its overlap check.
export const fitH = (w: number, label: string, sub?: string, icon?: string) => {
  const tw = (w - 48 - (icon ? 74 : 0)) * 0.92, lines = (s: string, px: number) => Math.max(1, Math.ceil((s.length * px) / tw));
  return Math.max(100, 46 + 42 * lines(label, LABEL_PX * 0.545) + (sub ? 6 + 34 * lines(sub, SUB_PX * 0.525) : 0));
};

// A diagram box: optional icon on the left, label and sub on the part itself. Before its word: nothing (a dashed
// outline only with "scaffold": true; an empty slot reads as a missing picture).
const Node: React.FC<{ b: Box; label: string; sub?: string; icon?: string; on: boolean; s: number; dim: boolean; badge?: string; ghost?: boolean }> = ({ b, label, sub, icon, on, s, dim, badge, ghost }) => {
  const at: React.CSSProperties = { position: "absolute", left: b.x - b.w / 2, top: b.y - b.h / 2, width: b.w, height: b.h };
  if (s <= 0) return ghost ? <div style={{ ...at, ...ghostStyle() }} /> : null;
  return (
    <div style={{ ...at, ...boxStyle(on), display: "flex", alignItems: "center", justifyContent: icon ? "flex-start" : "center", gap: 20,
      padding: "0 24px", boxSizing: "border-box", opacity: Math.min(1, s) * (dim ? DIM : 1), transform: `scale(${0.86 + 0.14 * s})` }}>
      {badge && (
        <div style={{ position: "absolute", left: -20, top: -20, width: 48, height: 48, borderRadius: 24, display: "flex", alignItems: "center",
          justifyContent: "center", fontFamily: MONO, fontSize: 28, fontWeight: 700, background: on ? hi() : C.ink, color: on ? onHi() : C.bg }}>{badge}</div>
      )}
      {icon && <Icon name={icon} size={56} color={on ? hi() : C.dim} />}
      <div style={{ textAlign: icon ? "left" : "center", minWidth: 0 }}>
        <div style={{ fontSize: LABEL_PX, fontWeight: 800, lineHeight: 1.1, letterSpacing: -0.5 }}>{label}</div>
        {sub && <div style={{ fontSize: SUB_PX, fontWeight: 600, color: C.dim, marginTop: 6, lineHeight: 1.2 }}>{sub}</div>}
      </div>
    </div>
  );
};

// An arrow's label sits on the line, on a page-coloured pill that masks it.
const pillW = (text: string) => text.length * EDGE_PX * 0.55 + 24;
const EdgeLabel: React.FC<{ at: Pt; text: string; on: boolean; o: number }> = ({ at, text, on, o }) => (
  <div style={{ position: "absolute", left: at.x, top: at.y, transform: "translate(-50%, -50%)", padding: "3px 12px", borderRadius: 8,
    background: C.bg, fontSize: EDGE_PX, fontWeight: 700, color: on ? hi() : C.dim, whiteSpace: "nowrap", opacity: o }}>{text}</div>
);

// ---------- build ----------

type BNode = { id: string; label: string; sub?: string; x: number; y: number; w?: number; h?: number; icon?: string; cue?: Cue };
type BEdge = { id?: string; from: string; to: string; label?: string; cue?: Cue; dashed?: boolean; loop?: boolean | "up" | "down"; bend?: number };
const eid = (e: BEdge) => e.id ?? `${e.from}>${e.to}`;
export const nodeBox = (n: BNode): Box => ({ x: n.x, y: n.y, w: n.w ?? 400, h: n.h ?? fitH(n.w ?? 400, n.label, n.sub, n.icon) });

// Straight, bent (a quadratic, "bend" px off the straight line) or a loop under (or over) both boxes.
const edgePath = (e: BEdge, a: Box, b: Box): Poly => {
  if (e.loop) {
    const sg = e.loop === "up" ? -1 : 1, depth = e.bend ?? 110;
    const s = { x: a.x, y: a.y + sg * (a.h / 2 + GAP) }, t = { x: b.x, y: b.y + sg * (b.h / 2 + GAP) };
    const yy = (sg > 0 ? Math.max(a.y + a.h / 2, b.y + b.h / 2) : Math.min(a.y - a.h / 2, b.y - b.h / 2)) + sg * depth;
    return poly(sample(bez(s, { x: s.x, y: yy }, { x: t.x, y: yy }, t)));
  }
  const L = Math.hypot(b.x - a.x, b.y - a.y) || 1, k = e.bend ?? 0;
  const c = { x: (a.x + b.x) / 2 - ((b.y - a.y) / L) * k * 2, y: (a.y + b.y) / 2 + ((b.x - a.x) / L) * k * 2 };
  const s = exit(a, c.x, c.y, GAP), t = exit(b, c.x, c.y, GAP);
  if (!k) return poly([s, t]);
  return poly(sample((u) => ({ x: (1 - u) ** 2 * s.x + 2 * (1 - u) * u * c.x + u * u * t.x, y: (1 - u) ** 2 * s.y + 2 * (1 - u) * u * c.y + u * u * t.y })));
};

// One fit for a whole cont chain (the union of its boxes, arrows and labels), so carried-over parts never move.
const buildFit = (visuals: any[]): Fit => {
  const nodes = new Map<string, BNode>(), edges = new Map<string, BEdge>();
  visuals.forEach((v) => { (v.nodes ?? []).forEach((n: BNode) => nodes.set(n.id, n)); (v.edges ?? []).forEach((e: BEdge) => edges.set(eid(e), e)); });
  const box = Object.fromEntries([...nodes.values()].map((n) => [n.id, nodeBox(n)]));
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  const add = (x: number, y: number, hw = 0, hh = 0) => { x0 = Math.min(x0, x - hw); x1 = Math.max(x1, x + hw); y0 = Math.min(y0, y - hh); y1 = Math.max(y1, y + hh); };
  Object.values(box).forEach((b) => add(b.x, b.y, b.w / 2 + 22, b.h / 2 + 22)); // + the halo and a step badge
  edges.forEach((e) => {
    if (!box[e.from] || !box[e.to]) return;
    const pl = edgePath(e, box[e.from], box[e.to]);
    pl.pts.forEach((p) => add(p.x, p.y, 12, 12));
    if (e.label) { const m = along(pl, pl.len / 2); add(m.x, m.y, pillW(e.label) / 2, 24); }
  });
  return x0 === Infinity ? { s: 1, tx: 0, ty: 0 } : fitBox(x0, y0, x1, y1);
};

type BState = { on: string[]; dim: boolean };
const buildPlan = (p: DP, fps: number) => {
  const v = p.v, c = cuer(p, fps);
  const nodes: BNode[] = v.nodes ?? [], edges: BEdge[] = v.edges ?? [], focus: { ids: string[]; cue?: Cue }[] = v.focus ?? [];
  const nAt = firstIn(revealAt(nodes, p, fps).map((a) => settle(p, a)));
  const atOf = Object.fromEntries(nodes.map((n, i) => [n.id, nAt[i]]));
  const eAt = edges.map((e) => { // an arrow never draws before both its boxes are in
    const ends = Math.max(atOf[e.from] ?? 0, atOf[e.to] ?? 0);
    return settle(p, Math.max(c(e.cue, ends + 8), ends));
  });
  const fAt = focus.map((f) => c(f.cue, Math.round(p.pre + p.len * 0.7)));
  // The latest event sets the one highlight: the part just revealed, or a focus step (which also dims the rest).
  const evs = [
    ...nodes.map((n, i) => ({ at: nAt[i], on: [n.id], dim: false })),
    ...edges.map((e, j) => ({ at: eAt[j], on: [eid(e)], dim: false })),
    ...focus.map((f, k) => ({ at: fAt[k], on: f.ids, dim: f.ids.length > 0 })),
  ].filter((e) => e.at !== SETTLED).sort((a, b) => a.at - b.at);
  return { nodes, edges, nAt, eAt, evs };
};
const stateAt = (evs: { at: number; on: string[]; dim: boolean }[], frame: number): BState | undefined => evs.filter((e) => e.at <= frame).pop();
// The highlight a beat ends on; a cont beat keeps it until its own first reveal or focus.
const finalState = (p: DP, fps: number): BState | undefined =>
  stateAt(buildPlan(p, fps).evs, p.pre + p.len - 1) ?? (p.v.cont && p.prev ? finalState(p.prev, fps) : undefined);

export const Build: React.FC<DP> = (p) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const { nodes, edges, nAt, eAt, evs } = buildPlan(p, fps);
  const fit = buildFit(p.chain ?? [p.v]);
  const box = Object.fromEntries(nodes.map((n) => [n.id, nodeBox(n)]));
  const ev = stateAt(evs, frame) ?? (p.v.cont && p.prev ? finalState(p.prev, fps) : undefined);
  const on = (id: string) => !!ev?.on.includes(id);
  const dimmed = (id: string, ends?: [string, string]) => !!ev?.dim && !on(id) && !(ends && on(ends[0]) && on(ends[1]));
  const paths = edges.map((e) => (box[e.from] && box[e.to] ? edgePath(e, box[e.from], box[e.to]) : null));
  return (
    <Staged fit={fit}>
      <svg style={{ position: "absolute", left: 0, top: 0, overflow: "visible" }} width={STAGE.w} height={STAGE.h}>
        {edges.map((e, j) => {
          const pl = paths[j];
          if (!pl) return null;
          const lit = on(eid(e));
          return (
            <g key={j} opacity={dimmed(eid(e), [e.from, e.to]) ? DIM : 1}>
              <Arrow pl={pl} p={ease(frame, eAt[j], 16)} color={lit ? hi() : ink2()} width={lit ? 5 : 4} dashed={e.dashed} id={`dd${p.index}e${j}`} />
            </g>
          );
        })}
      </svg>
      {edges.map((e, j) => {
        const pl = paths[j];
        if (!pl || !e.label) return null;
        const q = ease(frame, eAt[j], 16);
        return <EdgeLabel key={j} at={along(pl, pl.len / 2)} text={e.label} on={on(eid(e))}
          o={Math.max(0, (q - 0.5) * 2) * (dimmed(eid(e), [e.from, e.to]) ? DIM : 1)} />;
      })}
      {nodes.map((n, i) => (
        <Node key={n.id} b={box[n.id]} label={n.label} sub={n.sub} icon={n.icon} on={on(n.id)} dim={dimmed(n.id)}
          s={frame >= nAt[i] ? pop(frame, nAt[i], fps) : 0} ghost={p.v.scaffold === true} />
      ))}
    </Staged>
  );
};

// ---------- loop / steps ----------

type CNode = { label: string; sub?: string; icon?: string; cue?: Cue };

// loop: boxes on a ring, arrows along it, the last one closing the cycle. steps: a row of numbered boxes; "back" adds a
// return arrow under the row. On cycleCue a marker travels the cycle once and lights each box as it passes.
const cycleLayout = (v: any, ring: boolean) => {
  const nodes: CNode[] = v.nodes ?? [];
  const n = Math.max(1, nodes.length);
  const cx = 840, cy = 318, rx = n <= 3 ? 470 : 560, ry = 222;
  const th = (i: number) => -Math.PI / 2 + (2 * Math.PI * i) / n;
  const w = ring ? 360 : Math.min(400, (STAGE.w - 90 * (n - 1)) / n), gapX = ring ? 0 : Math.min(140, (STAGE.w - n * w) / Math.max(1, n - 1));
  const h = Math.max(ring ? 100 : 130, ...nodes.map((x) => fitH(w, x.label, x.sub, x.icon))); // one height for the set
  const rowW = n * w + (n - 1) * gapX, rowY = 280;
  const boxes: Box[] = nodes.map((_, i) => ring
    ? { x: cx + rx * Math.cos(th(i)), y: cy + ry * Math.sin(th(i)), w, h }
    : { x: 840 - rowW / 2 + w / 2 + i * (w + gapX), y: rowY, w, h });
  const ell = (t: number): Pt => ({ x: cx + rx * Math.cos(t), y: cy + ry * Math.sin(t) });
  // connector i runs from box i to box i+1; with a ring (or "back") the last one returns to box 0
  const arc = (i: number): Poly => {
    const j = (i + 1) % n, a = th(i), b = i + 1 < n ? th(i + 1) : th(0) + 2 * Math.PI;
    return poly(sample((t) => ell(a + (b - a) * t), 72).filter((q) => !inside(boxes[i], q, GAP) && !inside(boxes[j], q, GAP)));
  };
  const backY = rowY + h / 2 + 92;
  const conns: Poly[] = nodes.length < 2 ? [] : ring ? nodes.map((_, i) => arc(i)) : [
    ...nodes.slice(0, -1).map((_, i) => poly([{ x: boxes[i].x + w / 2 + GAP, y: rowY }, { x: boxes[i + 1].x - w / 2 - GAP, y: rowY }])),
    ...(v.back ? [poly([{ x: boxes[n - 1].x, y: rowY + h / 2 + GAP }, { x: boxes[n - 1].x, y: backY }, { x: boxes[0].x, y: backY },
      { x: boxes[0].x, y: rowY + h / 2 + GAP }])] : []),
  ];
  const route: Poly = ring ? poly(sample((t) => ell(th(0) + 2 * Math.PI * t), 120))
    : poly([...boxes.map((b) => ({ x: b.x, y: b.y })), ...(v.back ? [{ x: boxes[n - 1].x, y: backY }, { x: boxes[0].x, y: backY }, { x: boxes[0].x, y: rowY }] : [])]);
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  boxes.forEach((b) => { x0 = Math.min(x0, b.x - b.w / 2 - 22); x1 = Math.max(x1, b.x + b.w / 2 + 22); y0 = Math.min(y0, b.y - b.h / 2 - 22); y1 = Math.max(y1, b.y + b.h / 2 + 22); });
  if (!ring && v.back) y1 = Math.max(y1, backY + 24);
  return { nodes, n, boxes, conns, route, backY, cx, cy, w, fit: x0 === Infinity ? { s: 1, tx: 0, ty: 0 } : fitBox(x0, y0, x1, y1) };
};

export const Cycle: React.FC<DP & { ring: boolean }> = ({ ring, ...p }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const v = p.v, c = cuer(p, fps);
  const { nodes, n, boxes, conns, route, backY, cx, cy, fit } = cycleLayout(v, ring);
  const at = firstIn(revealAt(nodes, p, fps).map((a) => settle(p, a)));
  const closeAt = settle(p, c(v.back?.cue, (at[n - 1] ?? 0) + 10));
  const cAt = conns.map((_, i) => (i < n - 1 ? at[i + 1] : closeAt));
  const T = Math.round((v.cycleS ?? 0.55 * n) * fps);
  // the marker sets off on its cue, early enough to finish by the hold
  const c0 = v.cycleCue != null ? Math.max(0, Math.min(c(v.cycleCue, -1), p.hold - T)) : -1;
  const tc = c0 >= 0 ? (frame - c0) / T : -1;
  const moving = tc >= 0 && tc <= 1;
  const k = moving ? tc : 0;
  const eased = k < 0.5 ? 2 * k * k : 1 - (-2 * k + 2) ** 2 / 2;
  const mk = along(route, route.len * eased);
  const near = moving ? boxes.reduce((b, bx, i) => (Math.hypot(bx.x - mk.x, bx.y - mk.y) < Math.hypot(boxes[b].x - mk.x, boxes[b].y - mk.y) ? i : b), 0) : -1;
  const fresh = newest(at.map((a) => (a === SETTLED ? Infinity : a)), frame);
  const lit = moving ? near : tc > 1 ? -1 : fresh;
  const markerO = moving ? 1 : tc > 1 ? Math.max(0, 1 - (frame - c0 - T) / 6) : 0;
  const centerAt = c(v.centerCue, (at[n - 1] ?? 0) + 12);
  const backLabel = v.back?.label as string | undefined;
  return (
    <Staged fit={fit}>
      <svg style={{ position: "absolute", left: 0, top: 0, overflow: "visible" }} width={STAGE.w} height={STAGE.h}>
        {conns.map((pl, i) => <Arrow key={i} pl={pl} p={ease(frame, cAt[i], 14)} color={ink2()} id={`dd${p.index}c${i}`} />)}
        {markerO > 0 && <circle cx={mk.x} cy={mk.y} r={14} fill={hi()} stroke={C.bg} strokeWidth={5} opacity={markerO} />}
      </svg>
      {ring && v.center && (
        <div style={{ position: "absolute", left: cx - 300, width: 600, top: cy, transform: "translateY(-50%)", textAlign: "center",
          fontSize: 34, fontWeight: 700, color: C.dim, lineHeight: 1.2, opacity: ease(frame, centerAt, 10) }}>{v.center}</div>
      )}
      {backLabel && conns.length > 0 && (() => {
        const pl = conns[conns.length - 1];
        return <EdgeLabel at={ring ? along(pl, pl.len / 2) : { x: (boxes[0].x + boxes[n - 1].x) / 2, y: backY }} text={backLabel} on={false}
          o={Math.max(0, (ease(frame, closeAt, 14) - 0.5) * 2)} />;
      })()}
      {nodes.map((nd, i) => (
        <Node key={i} b={boxes[i]} label={nd.label} sub={nd.sub} icon={nd.icon} on={i === lit} dim={false} ghost={v.scaffold === true}
          s={frame >= at[i] ? pop(frame, at[i], fps) : 0} badge={ring ? undefined : String(i + 1)} />
      ))}
    </Staged>
  );
};

// ---------- timeline ----------

type TItem = { year: string; label: string; sub?: string; cue?: Cue; now?: boolean };

// A lineage on a horizontal axis that inks in up to each item as it's named (the first is in place at the cut). The
// newest item is lit; when the current work (the "now" item, else the last) lands, it takes the highlight and the
// earlier ones step back.
export const Timeline: React.FC<DP> = (p) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const items: TItem[] = p.v.items ?? [];
  const n = items.length;
  const at = firstIn(revealAt(items, p, fps).map((a) => settle(p, a)));
  const nowIdx = items.some((it) => it.now) ? items.findIndex((it) => it.now) : n - 1;
  const k = newest(at, frame);
  const nowOn = frame >= at[nowIdx];
  const x0 = 30, x1 = STAGE.w - 30, step = (x1 - x0 - 40) / n, ax = 280;
  const xi = (i: number) => x0 + step * (i + 0.5);
  const endOf = (i: number) => (i < 0 ? x0 : i === n - 1 ? x1 : xi(i) + step / 2);
  const prev = k < 0 ? x0 : endOf(newest(at.map((a, i) => (i === k ? Infinity : a)), frame));
  const end = k < 0 ? x0 : prev + (endOf(k) - prev) * ease(frame, at[k], 14);
  const axis = poly([{ x: x0, y: ax }, { x: x1, y: ax }]);
  return (
    <>
      <svg style={{ position: "absolute", left: STAGE.x, top: STAGE.y, overflow: "visible" }} width={STAGE.w} height={STAGE.h}>
        {p.v.scaffold === true && <line x1={x0} y1={ax} x2={x1} y2={ax} stroke={line()} strokeWidth={3} strokeDasharray="10 10" />}
        <Arrow pl={axis} p={(end - x0) / (x1 - x0)} color={ink2()} id={`dd${p.index}t`} />
      </svg>
      {items.map((it, i) => {
        const s = frame >= at[i] ? pop(frame, at[i], fps) : 0;
        const lit = nowOn ? i === nowIdx : i === k;
        const r = i === nowIdx ? 18 : 13;
        return (
          <div key={i} style={{ opacity: Math.min(1, s) * (nowOn && i !== nowIdx ? 0.5 : 1) }}>
            <div style={{ position: "absolute", left: STAGE.x + xi(i) - r, top: STAGE.y + ax - r, width: r * 2, height: r * 2, borderRadius: r,
              boxSizing: "border-box", background: lit ? hi() : C.bg, border: `4px solid ${lit ? hi() : current.light ? C.ink : "#f5f4f2"}`,
              transform: `scale(${s})` }} />
            <div style={{ position: "absolute", left: STAGE.x + xi(i) - step / 2, width: step, top: STAGE.y + ax - 90, textAlign: "center",
              fontFamily: MONO, fontSize: 42, fontWeight: 600, color: lit ? hi() : C.dim }}>{it.year}</div>
            <div style={{ position: "absolute", left: STAGE.x + xi(i) - step / 2 + 10, width: step - 20, top: STAGE.y + ax + 42, textAlign: "center",
              transform: `translateY(${(1 - Math.min(1, s)) * 16}px)` }}>
              <div style={{ fontSize: LABEL_PX, fontWeight: 800, lineHeight: 1.12, letterSpacing: -0.4 }}>{it.label}</div>
              {it.sub && <div style={{ fontSize: SUB_PX, fontWeight: 600, color: C.dim, marginTop: 8, lineHeight: 1.25 }}>{it.sub}</div>}
            </div>
          </div>
        );
      })}
    </>
  );
};

// ---------- bars ----------

type Bar = { label: string; value?: number; display?: string; cue?: Cue; note?: string; range?: [number, number]; denominator?: string };

// A bar chart, vertical (default) or horizontal: the first bar is in place at the cut and each next one grows on its
// word, with its category and its number at the bar's end. range: a low-high span ("20 to 60 percent"); denominator: the
// count behind a share ("57 of 86"); baseline: a dashed reference line. The newest bar carries the highlight.
export const Bars: React.FC<DP> = (p) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const v = p.v, c = cuer(p, fps);
  const bars: Bar[] = v.bars ?? [];
  const n = Math.max(1, bars.length), horiz = v.orient === "h";
  const top = (b: Bar) => (b.range ? b.range[1] : b.value ?? 0);
  const pct = bars.some((b) => /%/.test(b.display ?? ""));
  const max: number = v.max ?? (pct && bars.every((b) => top(b) <= 100) ? 100 : Math.max(1, ...bars.map(top), v.baseline?.value ?? 0) * 1.12);
  const at = firstIn(revealAt(bars, p, fps).map((a) => settle(p, a)));
  const k = newest(at, frame);
  const bAt = v.baseline ? c(v.baseline.cue, p.born < 0 ? SETTLED : 0) : 0;
  const titleH = v.title ? 64 : 0;
  // plot geometry (stage-local): value axis runs from `zero` to `full`
  // (vertical charts keep a right margin for the baseline's label, clear of the bars' numbers)
  const L = horiz ? { x0: 470, x1: STAGE.w - 230, y0: titleH + 20, y1: STAGE.h - 20 }
    : { x0: 60, x1: STAGE.w - (v.baseline?.label ? 280 : 60), y0: titleH + 120, y1: STAGE.h - 130 };
  const slot = horiz ? Math.min(150, (L.y1 - L.y0) / n) : Math.min(380, (L.x1 - L.x0) / n);
  const thick = horiz ? Math.min(66, slot * 0.5) : Math.min(170, slot * 0.5);
  const centre = (i: number) => (horiz ? (L.y0 + L.y1) / 2 : (L.x0 + L.x1) / 2) + (i - (n - 1) / 2) * slot;
  const pos = (val: number) => (horiz ? L.x0 + ((L.x1 - L.x0) * val) / max : L.y1 - ((L.y1 - L.y0) * val) / max);
  const ink = (lit: boolean) => (lit ? hi() : current.light ? "#8b8d93" : "#77767a");
  const abs = (x: number, y: number) => ({ left: STAGE.x + x, top: STAGE.y + y });
  return (
    <>
      {v.title && <div style={{ position: "absolute", ...abs(horiz ? 0 : L.x0, 0), width: STAGE.w, fontSize: 38, fontWeight: 800, letterSpacing: -0.5,
        opacity: ease(frame, p.born < 0 ? SETTLED : 0, 10) }}>{v.title}</div>}
      <svg style={{ position: "absolute", ...abs(0, 0), overflow: "visible" }} width={STAGE.w} height={STAGE.h}>
        {v.scaffold === true && bars.map((b, i) => { // empty tracks, full scale, only on request
          const cc = centre(i);
          return horiz
            ? <rect key={i} x={L.x0} y={cc - thick / 2} width={L.x1 - L.x0} height={thick} rx={8} fill={line()} opacity={0.35} />
            : <rect key={i} x={cc - thick / 2} y={L.y0} width={thick} height={L.y1 - L.y0} rx={8} fill={line()} opacity={0.35} />;
        })}
        {bars.map((b, i) => {
          const g = ease(frame, at[i], 18), lit = i === k, cc = centre(i);
          if (g <= 0) return null;
          const lo = b.range ? b.range[0] : 0, hiV = lo + (top(b) - lo) * g;
          const a = pos(lo), z = pos(hiV);
          const r = horiz ? { x: Math.min(a, z), y: cc - thick / 2, w: Math.abs(z - a), h: thick } : { x: cc - thick / 2, y: Math.min(a, z), w: thick, h: Math.abs(z - a) };
          return b.range
            ? <rect key={i} x={r.x} y={r.y} width={r.w} height={r.h} rx={8} fill={lit ? hi() : ink(false)} fillOpacity={0.28} stroke={ink(lit)} strokeWidth={4} strokeDasharray="12 7" />
            : <rect key={i} x={r.x} y={r.y} width={r.w} height={r.h} rx={8} fill={ink(lit)} />;
        })}
        {horiz ? <line x1={L.x0} y1={L.y0 - 10} x2={L.x0} y2={L.y1 + 10} stroke={ink2()} strokeWidth={3} />
          : <line x1={L.x0} y1={L.y1} x2={L.x1} y2={L.y1} stroke={ink2()} strokeWidth={3} />}
        {v.baseline && (() => {
          const q = pos(v.baseline.value), o = ease(frame, bAt, 10);
          return horiz ? <line x1={q} y1={L.y0 - 14} x2={q} y2={L.y1 + 14} stroke={ink2()} strokeWidth={3} strokeDasharray="10 8" opacity={o} />
            : <line x1={L.x0} y1={q} x2={L.x1} y2={q} stroke={ink2()} strokeWidth={3} strokeDasharray="10 8" opacity={o} />;
        })()}
      </svg>
      {v.baseline?.label && (() => {
        const q = pos(v.baseline.value);
        return <div style={{ position: "absolute", ...(horiz ? abs(q + 10, L.y1 - 10) : abs(L.x1 + 16, q)), width: horiz ? 300 : 250,
          transform: horiz ? "none" : "translateY(-50%)", fontSize: SUB_PX, fontWeight: 700, color: C.dim, lineHeight: 1.15,
          opacity: ease(frame, bAt, 10) }}>{v.baseline.label}</div>;
      })()}
      {bars.map((b, i) => {
        const g = ease(frame, at[i], 18), lit = i === k, cc = centre(i), shown = frame >= at[i];
        const o = interpolate(g, [0.55, 1], [0, 1], clamp);
        const tip = pos((b.range ? b.range[0] : 0) + (top(b) - (b.range ? b.range[0] : 0)) * g);
        const text = b.display ?? (b.range ? `${b.range[0]}–${b.range[1]}` : `${b.value ?? ""}`);
        const val = <span style={{ fontSize: horiz ? 52 : 60, fontWeight: 900, letterSpacing: -1.5, color: lit ? hi() : C.ink, fontVariantNumeric: "tabular-nums" }}>{text}</span>;
        const den = b.denominator && <span style={{ fontSize: SUB_PX, fontWeight: 700, color: C.dim }}>{b.denominator}</span>;
        return (
          <React.Fragment key={i}>
            {horiz ? (
              <>
                <div style={{ position: "absolute", ...abs(0, cc), width: L.x0 - 30, transform: "translateY(-50%)", textAlign: "right",
                  opacity: shown ? 1 : v.scaffold === true ? 0.45 : 0 }}>
                  <div style={{ fontSize: 36, fontWeight: 800, lineHeight: 1.1 }}>{b.label}</div>
                  {b.note && <div style={{ fontSize: SUB_PX, fontWeight: 600, color: C.dim, marginTop: 4, opacity: o }}>{b.note}</div>}
                </div>
                <div style={{ position: "absolute", ...abs(tip + 18, cc), transform: "translateY(-50%)", display: "flex", alignItems: "baseline",
                  gap: 14, whiteSpace: "nowrap", opacity: o }}>{val}{den}</div>
              </>
            ) : (
              <>
                <div style={{ position: "absolute", ...abs(cc - slot / 2, tip - 12), width: slot, transform: "translateY(-100%)", textAlign: "center",
                  display: "flex", flexDirection: "column", alignItems: "center", opacity: o }}>{val}{den}</div>
                <div style={{ position: "absolute", ...abs(cc - slot / 2 + 8, L.y1 + 16), width: slot - 16, textAlign: "center",
                  opacity: shown ? 1 : v.scaffold === true ? 0.45 : 0 }}>
                  <div style={{ fontSize: 36, fontWeight: 800, lineHeight: 1.1 }}>{b.label}</div>
                  {b.note && <div style={{ fontSize: SUB_PX, fontWeight: 600, color: C.dim, marginTop: 6, lineHeight: 1.2, opacity: o }}>{b.note}</div>}
                </div>
              </>
            )}
          </React.Fragment>
        );
      })}
    </>
  );
};
