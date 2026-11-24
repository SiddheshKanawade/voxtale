import React from "react";
import { useCurrentFrame, useVideoConfig } from "remotion";
import { C, STAGE } from "../theme";
import { current } from "../design";
import {
  Arrow, Box, Cue, DIM, DP, SETTLED, Icon, MONO, Poly, Pt, along, bez, boxStyle, cueAt, ease, exit, hi, inside, ink2, newest, onHi,
  poly, pop, revealAt, sample, settle,
} from "./kit";

// Our own diagrams, built step by step with the narration: build (generic boxes and arrows), loop / steps (an
// iterative process), timeline (a lineage). Coordinates are stage-local: (0, 0) is STAGE's top-left, 1680 x 660.

const GAP = 10; // between a box and the arrow that meets it

// A diagram box: optional icon on the left, label and sub on the part itself.
const Node: React.FC<{ b: Box; label: string; sub?: string; icon?: string; on: boolean; s: number; dim: boolean; badge?: string }> = ({ b, label, sub, icon, on, s, dim, badge }) => (
  <div style={{ position: "absolute", left: STAGE.x + b.x - b.w / 2, top: STAGE.y + b.y - b.h / 2, width: b.w, height: b.h, ...boxStyle(on),
    display: "flex", alignItems: "center", justifyContent: icon ? "flex-start" : "center", gap: 18, padding: "0 24px",
    boxSizing: "border-box", opacity: s * (dim ? DIM : 1), transform: `scale(${0.86 + 0.14 * s})` }}>
    {badge && (
      <div style={{ position: "absolute", left: -18, top: -18, width: 44, height: 44, borderRadius: 22, display: "flex", alignItems: "center",
        justifyContent: "center", fontFamily: MONO, fontSize: 22, fontWeight: 700, background: on ? hi() : C.ink, color: on ? onHi() : C.bg }}>{badge}</div>
    )}
    {icon && <Icon name={icon} size={54} color={on ? hi() : C.dim} />}
    <div style={{ textAlign: icon ? "left" : "center", minWidth: 0 }}>
      <div style={{ fontSize: 34, fontWeight: 800, lineHeight: 1.08, letterSpacing: -0.5 }}>{label}</div>
      {sub && <div style={{ fontSize: 24, fontWeight: 600, color: C.dim, marginTop: 6, lineHeight: 1.2 }}>{sub}</div>}
    </div>
  </div>
);

// An arrow's label sits on the line, on a page-coloured pill that masks it.
const EdgeLabel: React.FC<{ at: Pt; text: string; on: boolean; o: number }> = ({ at, text, on, o }) => (
  <div style={{ position: "absolute", left: STAGE.x + at.x, top: STAGE.y + at.y, transform: "translate(-50%, -50%)", padding: "3px 12px",
    borderRadius: 8, background: C.bg, fontSize: 25, fontWeight: 700, color: on ? hi() : C.dim, whiteSpace: "nowrap", opacity: o }}>{text}</div>
);

// ---------- build ----------

type BNode = { id: string; label: string; sub?: string; x: number; y: number; w?: number; h?: number; icon?: string; cue?: Cue };
type BEdge = { id?: string; from: string; to: string; label?: string; cue?: Cue; dashed?: boolean; loop?: boolean | "up" | "down"; bend?: number };
const eid = (e: BEdge) => e.id ?? `${e.from}>${e.to}`;
// Box height for its text (34 px label, 24 px sub; bold glyphs about 0.55 em wide), so nothing spills out of a box.
// tools/deepdive/dd.py mirrors it for its overlap check.
export const fitH = (w: number, label: string, sub?: string, icon?: string) => {
  const tw = (w - 48 - (icon ? 72 : 0)) * 0.92, lines = (s: string, px: number) => Math.max(1, Math.ceil((s.length * px) / tw));
  return Math.max(96, 44 + 38 * lines(label, 18.5) + (sub ? 6 + 29 * lines(sub, 12.6) : 0));
};
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

export const Build: React.FC<DP> = ({ v, dur, index, words }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const nodes: BNode[] = v.nodes ?? [], edges: BEdge[] = v.edges ?? [], focus: { ids: string[]; cue?: Cue }[] = v.focus ?? [];
  const box = Object.fromEntries(nodes.map((n) => [n.id, nodeBox(n)]));
  const nAt = revealAt(nodes, words, fps, dur).map((a) => settle(v, a));
  const atOf = Object.fromEntries(nodes.map((n, i) => [n.id, nAt[i]]));
  const eAt = edges.map((e) => { // an arrow never draws before both its boxes are in
    const ends = Math.max(atOf[e.from] ?? 0, atOf[e.to] ?? 0);
    return settle(v, Math.max(cueAt(words, e.cue, fps, ends + 8), ends));
  });
  const fAt = focus.map((f) => cueAt(words, f.cue, fps, Math.round(dur * 0.7)));
  // The latest event sets the one highlight: the part just revealed, or a focus step (which also dims the rest).
  const evs = [
    ...nodes.map((n, i) => ({ at: nAt[i], on: [n.id], dim: false })),
    ...edges.map((e, j) => ({ at: eAt[j], on: [eid(e)], dim: false })),
    ...focus.map((f, k) => ({ at: fAt[k], on: f.ids, dim: f.ids.length > 0 })),
  ].filter((e) => e.at <= frame && e.at !== SETTLED).sort((a, b) => a.at - b.at);
  const ev = evs[evs.length - 1];
  const on = (id: string) => !!ev?.on.includes(id);
  const dimmed = (id: string, ends?: [string, string]) => !!ev?.dim && !on(id) && !(ends && on(ends[0]) && on(ends[1]));
  const paths = edges.map((e) => (box[e.from] && box[e.to] ? edgePath(e, box[e.from], box[e.to]) : null));
  return (
    <>
      <svg style={{ position: "absolute", left: STAGE.x, top: STAGE.y, overflow: "visible" }} width={STAGE.w} height={STAGE.h}>
        {edges.map((e, j) => {
          const pl = paths[j];
          if (!pl) return null;
          const lit = on(eid(e));
          return (
            <g key={j} opacity={dimmed(eid(e), [e.from, e.to]) ? DIM : 1}>
              <Arrow pl={pl} p={ease(frame, eAt[j], 16)} color={lit ? hi() : ink2()} width={lit ? 5 : 4} dashed={e.dashed} id={`dd${index}e${j}`} />
            </g>
          );
        })}
      </svg>
      {edges.map((e, j) => {
        const pl = paths[j];
        if (!pl || !e.label) return null;
        const p = ease(frame, eAt[j], 16);
        return <EdgeLabel key={j} at={along(pl, pl.len / 2)} text={e.label} on={on(eid(e))}
          o={Math.max(0, (p - 0.5) * 2) * (dimmed(eid(e), [e.from, e.to]) ? DIM : 1)} />;
      })}
      {nodes.map((n, i) => (
        <Node key={n.id} b={box[n.id]} label={n.label} sub={n.sub} icon={n.icon} on={on(n.id)} dim={dimmed(n.id)}
          s={frame >= nAt[i] ? pop(frame, nAt[i], fps) : 0} />
      ))}
    </>
  );
};

// ---------- loop / steps ----------

type CNode = { label: string; sub?: string; icon?: string; cue?: Cue };

// loop: boxes on a ring, arrows along it, the last one closing the cycle. steps: a row of numbered boxes; "back" adds a
// return arrow under the row. On cycleCue a marker travels the cycle once and lights each box as it passes.
export const Cycle: React.FC<DP & { ring: boolean }> = ({ v, dur, index, words, ring }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const nodes: CNode[] = v.nodes ?? [];
  const n = nodes.length;
  const cx = 840, cy = 318, rx = n <= 3 ? 470 : 560, ry = 222;
  const th = (i: number) => -Math.PI / 2 + (2 * Math.PI * i) / n;
  const w = ring ? 340 : Math.min(360, (STAGE.w - 80 * (n - 1)) / n), gapX = ring ? 0 : (STAGE.w - n * w) / Math.max(1, n - 1);
  const h = Math.max(ring ? 92 : 120, ...nodes.map((x) => fitH(w, x.label, x.sub, x.icon))); // one height for the set
  const rowY = v.back ? 250 : 318;
  const boxes: Box[] = nodes.map((_, i) => ring
    ? { x: cx + rx * Math.cos(th(i)), y: cy + ry * Math.sin(th(i)), w, h }
    : { x: w / 2 + i * (w + gapX), y: rowY, w, h });
  const ell = (t: number): Pt => ({ x: cx + rx * Math.cos(t), y: cy + ry * Math.sin(t) });
  // connector i runs from box i to box i+1; with a ring (or "back") the last one returns to box 0
  const arc = (i: number): Poly => {
    const j = (i + 1) % n, a = th(i), b = i + 1 < n ? th(i + 1) : th(0) + 2 * Math.PI;
    const pts = sample((t) => ell(a + (b - a) * t), 72).filter((p) => !inside(boxes[i], p, GAP) && !inside(boxes[j], p, GAP));
    return poly(pts);
  };
  const backY = rowY + h / 2 + 92;
  const conns: Poly[] = ring ? nodes.map((_, i) => arc(i)) : [
    ...nodes.slice(0, -1).map((_, i) => poly([{ x: boxes[i].x + w / 2 + GAP, y: rowY }, { x: boxes[i + 1].x - w / 2 - GAP, y: rowY }])),
    ...(v.back ? [poly([{ x: boxes[n - 1].x, y: rowY + h / 2 + GAP }, { x: boxes[n - 1].x, y: backY }, { x: boxes[0].x, y: backY },
      { x: boxes[0].x, y: rowY + h / 2 + GAP }])] : []),
  ];
  const at = revealAt(nodes, words, fps, dur).map((a) => settle(v, a));
  const closeAt = settle(v, cueAt(words, v.back?.cue, fps, (at[n - 1] ?? 0) + 10));
  const cAt = conns.map((_, i) => (i < n - 1 ? at[i + 1] : closeAt));
  // the marker's route: the ring itself, or through the row's centres (and back along the return arrow)
  const route: Poly = ring ? poly(sample((t) => ell(th(0) + 2 * Math.PI * t), 120))
    : poly([...boxes.map((b) => ({ x: b.x, y: b.y })), ...(v.back ? [{ x: boxes[n - 1].x, y: backY }, { x: boxes[0].x, y: backY }, { x: boxes[0].x, y: rowY }] : [])]);
  const c0 = v.cycleCue != null ? cueAt(words, v.cycleCue, fps, -1) : -1;
  const T = Math.round((v.cycleS ?? 0.55 * n) * fps);
  const tc = c0 >= 0 ? (frame - c0) / T : -1;
  const moving = tc >= 0 && tc <= 1;
  const k = tc < 0 || tc > 1 ? -1 : Math.min(1, Math.max(0, tc));
  const eased = k < 0 ? 0 : k < 0.5 ? 2 * k * k : 1 - (-2 * k + 2) ** 2 / 2;
  const mk = along(route, route.len * eased);
  const near = moving ? boxes.reduce((b, bx, i) => (Math.hypot(bx.x - mk.x, bx.y - mk.y) < Math.hypot(boxes[b].x - mk.x, boxes[b].y - mk.y) ? i : b), 0) : -1;
  const lit = moving ? near : tc > 1 ? -1 : newest(at.map((a) => (a === SETTLED ? Infinity : a)), frame);
  const markerO = moving ? 1 : tc > 1 ? Math.max(0, 1 - (frame - c0 - T) / 6) : 0;
  const centerAt = cueAt(words, v.centerCue, fps, (at[n - 1] ?? 0) + 12);
  const backLabel = v.back?.label as string | undefined;
  return (
    <>
      <svg style={{ position: "absolute", left: STAGE.x, top: STAGE.y, overflow: "visible" }} width={STAGE.w} height={STAGE.h}>
        {conns.map((pl, i) => <Arrow key={i} pl={pl} p={ease(frame, cAt[i], 14)} color={ink2()} id={`dd${index}c${i}`} />)}
        {markerO > 0 && <circle cx={mk.x} cy={mk.y} r={14} fill={hi()} stroke={C.bg} strokeWidth={5} opacity={markerO} />}
      </svg>
      {ring && v.center && (
        <div style={{ position: "absolute", left: STAGE.x + cx - 300, width: 600, top: STAGE.y + cy, transform: "translateY(-50%)",
          textAlign: "center", fontSize: 32, fontWeight: 700, color: C.dim, lineHeight: 1.2, opacity: ease(frame, centerAt, 10) }}>{v.center}</div>
      )}
      {backLabel && (() => {
        const pl = conns[conns.length - 1];
        return <EdgeLabel at={ring ? along(pl, pl.len / 2) : { x: (boxes[0].x + boxes[n - 1].x) / 2, y: backY }} text={backLabel} on={false}
          o={Math.max(0, (ease(frame, closeAt, 14) - 0.5) * 2)} />;
      })()}
      {nodes.map((nd, i) => (
        <Node key={i} b={boxes[i]} label={nd.label} sub={nd.sub} icon={nd.icon} on={i === lit} dim={false}
          s={frame >= at[i] ? pop(frame, at[i], fps) : 0} badge={ring ? undefined : String(i + 1)} />
      ))}
    </>
  );
};

// ---------- timeline ----------

type TItem = { year: string; label: string; sub?: string; cue?: Cue; now?: boolean };

// A lineage on a horizontal axis that extends to each item as it's named. The newest item is lit; when the current work
// (the "now" item, else the last) lands, it takes the highlight and the earlier ones step back.
export const Timeline: React.FC<DP> = ({ v, dur, index, words }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const items: TItem[] = v.items ?? [];
  const n = items.length;
  const at = revealAt(items, words, fps, dur).map((a) => settle(v, a));
  const nowIdx = items.some((it) => it.now) ? items.findIndex((it) => it.now) : n - 1;
  const k = newest(at, frame);
  const nowOn = frame >= at[nowIdx];
  const x0 = 30, x1 = STAGE.w - 30, step = (x1 - x0 - 40) / n, ax = 300;
  const xi = (i: number) => x0 + step * (i + 0.5);
  // the axis grows to just past the newest item, and to the end (an arrow: time runs right) with the last
  const endOf = (i: number) => (i < 0 ? x0 : i === n - 1 ? x1 : xi(i) + step / 2);
  const prev = k < 0 ? x0 : endOf(newest(at.map((a, i) => (i === k ? Infinity : a)), frame));
  const end = k < 0 ? x0 : prev + (endOf(k) - prev) * ease(frame, at[k], 14);
  const axis = poly([{ x: x0, y: ax }, { x: x1, y: ax }]);
  return (
    <>
      <svg style={{ position: "absolute", left: STAGE.x, top: STAGE.y, overflow: "visible" }} width={STAGE.w} height={STAGE.h}>
        <Arrow pl={axis} p={(end - x0) / (x1 - x0)} color={ink2()} id={`dd${index}t`} />
      </svg>
      {items.map((it, i) => {
        const s = frame >= at[i] ? pop(frame, at[i], fps) : 0;
        const lit = nowOn ? i === nowIdx : i === k;
        const big = i === nowIdx;
        const r = big ? 17 : 12;
        return (
          <div key={i} style={{ opacity: s * (nowOn && i !== nowIdx ? 0.5 : 1) }}>
            <div style={{ position: "absolute", left: STAGE.x + xi(i) - r, top: STAGE.y + ax - r, width: r * 2, height: r * 2, borderRadius: r,
              boxSizing: "border-box", background: lit ? hi() : C.bg, border: `4px solid ${lit ? hi() : current.light ? C.ink : "#f5f4f2"}`,
              transform: `scale(${s})` }} />
            <div style={{ position: "absolute", left: STAGE.x + xi(i) - step / 2, width: step, top: STAGE.y + ax - 82, textAlign: "center",
              fontFamily: MONO, fontSize: 36, fontWeight: 600, color: lit ? hi() : C.dim }}>{it.year}</div>
            <div style={{ position: "absolute", left: STAGE.x + xi(i) - step / 2 + 12, width: step - 24, top: STAGE.y + ax + 40, textAlign: "center",
              transform: `translateY(${(1 - s) * 16}px)` }}>
              <div style={{ fontSize: 34, fontWeight: 800, lineHeight: 1.12, letterSpacing: -0.4 }}>{it.label}</div>
              {it.sub && <div style={{ fontSize: 24, fontWeight: 600, color: C.dim, marginTop: 8, lineHeight: 1.25 }}>{it.sub}</div>}
            </div>
          </div>
        );
      })}
    </>
  );
};
