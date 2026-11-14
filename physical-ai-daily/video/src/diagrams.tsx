import React from "react";
import { AbsoluteFill, Easing, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { C, Word } from "./theme";
import { current } from "./design";
import { cueFrame } from "./scenes";

const clamp = { extrapolateLeft: "clamp", extrapolateRight: "clamp" } as const;

// Kitting: mixed parts in bulk bins get picked, in order, into the slots of a kit tray.
type Shape = "circle" | "square" | "tri";
const COLORS: Record<Shape, string> = { circle: "#fb923c", square: "#2dd4bf", tri: "#f472b6" };
const BIN_PARTS: { shape: Shape; dx: number; dy: number }[][] = [
  [{ shape: "circle", dx: -70, dy: 10 }, { shape: "tri", dx: 0, dy: -20 }, { shape: "square", dx: 70, dy: 15 }, { shape: "circle", dx: 30, dy: 55 }],
  [{ shape: "square", dx: -60, dy: -10 }, { shape: "circle", dx: 20, dy: 20 }, { shape: "tri", dx: 75, dy: -15 }, { shape: "square", dx: -20, dy: 55 }],
  [{ shape: "tri", dx: -65, dy: 20 }, { shape: "square", dx: 10, dy: -15 }, { shape: "circle", dx: 70, dy: 25 }, { shape: "tri", dx: 5, dy: 60 }],
];
const BINS = [{ x: 330, y: 640 }, { x: 610, y: 640 }, { x: 890, y: 640 }];
const SLOTS: { shape: Shape; from: [number, number] }[] = [
  { shape: "circle", from: [1, 1] }, { shape: "tri", from: [0, 1] }, { shape: "square", from: [2, 1] }, { shape: "circle", from: [0, 0] },
];
const slotPos = (i: number) => ({ x: 1250 + (i % 2) * 230, y: 470 + Math.floor(i / 2) * 210 });

const Part: React.FC<{ shape: Shape; x: number; y: number; r?: number; ghost?: boolean }> = ({ shape, x, y, r = 34, ghost }) => {
  const style = ghost ? { fill: "none", stroke: "rgba(255,255,255,0.35)", strokeWidth: 4, strokeDasharray: "8 6" } : { fill: COLORS[shape] };
  if (shape === "circle") return <circle cx={x} cy={y} r={r} {...style} />;
  if (shape === "square") return <rect x={x - r} y={y - r} width={r * 2} height={r * 2} rx={6} {...style} />;
  return <polygon points={`${x},${y - r * 1.1} ${x + r},${y + r * 0.8} ${x - r},${y + r * 0.8}`} {...style} />;
};

export const Kitting: React.FC<{ dur: number; words: Word[] }> = ({ dur, words }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const trayAt = cueFrame(words, "trays", fps, Math.round(dur * 0.35));
  const pickAt = cueFrame(words, "pick", fps, Math.round(dur * 0.3));
  const per = Math.max(12, Math.round((dur - pickAt - fps * 0.6) / SLOTS.length));
  const tray = spring({ frame: frame - trayAt, fps, config: { damping: 16 } });
  const binsIn = spring({ frame, fps, config: { damping: 16 } });
  const picked = (b: number, k: number) => SLOTS.some((s, i) => s.from[0] === b && s.from[1] === k && frame >= pickAt + i * per);
  return (
    <AbsoluteFill>
      <svg width={1920} height={1080}>
        <text x={610} y={470} fill={C.dim} fontSize={36} fontWeight={700} textAnchor="middle" opacity={binsIn}>Bulk bins: parts mixed together</text>
        {BINS.map((b, bi) => (
          <g key={bi} transform={`translate(0 ${(1 - binsIn) * 120})`}>
            {/* neutral bins (were saturated blue, off-palette): the coloured parts stay the only focal colour */}
            <rect x={b.x - 125} y={b.y - 100} width={250} height={200} rx={18} fill={C.panel2} stroke={C.dim} strokeOpacity={0.35} strokeWidth={3} />
            <rect x={b.x - 125} y={b.y - 100} width={250} height={36} rx={12} fill={C.dim} fillOpacity={0.3} />
            {BIN_PARTS[bi].map((p, k) => !picked(bi, k) && <Part key={k} shape={p.shape} x={b.x + p.dx} y={b.y + p.dy} r={26} />)}
          </g>
        ))}
        <g opacity={tray} transform={`translate(${(1 - tray) * 200} 0)`}>
          <text x={1365} y={360} fill={C.dim} fontSize={36} fontWeight={700} textAnchor="middle">Kit tray, in build order</text>
          <rect x={1110} y={380} width={510} height={440} rx={22} fill={C.panel2} stroke={C.accent} strokeWidth={4} />
          {SLOTS.map((s, i) => {
            const p = slotPos(i);
            return (
              <g key={i}>
                <rect x={p.x - 90} y={p.y - 80} width={180} height={170} rx={14} fill={current.light ? "rgba(31,32,35,0.06)" : "rgba(0,0,0,0.35)"}
                  stroke={C.dim} strokeOpacity={current.light ? 0.3 : 0} strokeDasharray="8 6" strokeWidth={2} />
                <text x={p.x - 74} y={p.y - 50} fill={C.accent} fontSize={30} fontWeight={900}>{i + 1}</text>
                <Part shape={s.shape} x={p.x} y={p.y + 5} ghost />
              </g>
            );
          })}
        </g>
        {SLOTS.map((s, i) => {
          const start = pickAt + i * per;
          if (frame < start) return null;
          const t = interpolate(frame, [start, start + per * 0.8], [0, 1], { ...clamp, easing: Easing.inOut(Easing.cubic) });
          const [bi, k] = s.from;
          const a = { x: BINS[bi].x + BIN_PARTS[bi][k].dx, y: BINS[bi].y + BIN_PARTS[bi][k].dy };
          const b = slotPos(i);
          const x = a.x + (b.x - a.x) * t, y = a.y + (b.y + 5 - a.y) * t - Math.sin(t * Math.PI) * 220;
          return (
            <g key={i}>
              <Part shape={s.shape} x={x} y={y} r={26 + 8 * t} />
              {t >= 1 && <text x={b.x + 50} y={b.y + 70} fontSize={44}>✅</text>}
            </g>
          );
        })}
      </svg>
    </AbsoluteFill>
  );
};

// Snap-through: light slowly loads a curved strip (energy meter fills), then it flips at once on the word "snaps".
export const Snap: React.FC<{ dur: number; words: Word[] }> = ({ dur, words }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const warmAt = cueFrame(words, "warms", fps, 0);
  const bendAt = cueFrame(words, "bends", fps, Math.round(dur * 0.2));
  const snapAt = cueFrame(words, "snaps", fps, Math.round(dur * 0.7));
  const load = interpolate(frame, [bendAt, snapAt], [0, 1], { ...clamp, easing: Easing.in(Easing.quad) });
  const snapped = frame >= snapAt;
  const flip = spring({ frame: frame - snapAt, fps, config: { damping: 7, stiffness: 260, mass: 0.5 } });
  const tremble = !snapped && load > 0.6 ? Math.sin(frame * 2.2) * 6 * load : 0;
  const h = snapped ? interpolate(flip, [0, 1], [70, -110]) : interpolate(load, [0, 1], [190, 70]) + tremble;
  const heat = interpolate(frame, [warmAt, bendAt + 20], [0, 1], clamp);
  const strip = snapped ? C.accent : `rgb(${130 + 120 * heat}, ${130 - 20 * heat}, ${140 - 100 * heat})`;
  const meter = snapped ? Math.max(0, 1 - (frame - snapAt) / 6) : load;
  const burst = spring({ frame: frame - snapAt, fps, config: { damping: 10 } });
  const phase = snapped ? 2 : frame >= bendAt ? 1 : 0;
  const labels = ["Light warms the material", "Strip bends, storing energy", "SNAP! Energy released at once"];
  const x0 = 620, x1 = 1300, y = 720;
  return (
    <AbsoluteFill>
      <svg width={1920} height={1080}>
        <g transform={`translate(330 330) rotate(${frame * 0.6})`}>
          {Array.from({ length: 12 }).map((_, i) => (
            <line key={i} x1={0} y1={-95} x2={0} y2={-135} stroke="#fcd34d" strokeWidth={10} strokeLinecap="round" transform={`rotate(${i * 30})`} />
          ))}
          <circle r={75} fill="#fcd34d" />
        </g>
        {Array.from({ length: 5 }).map((_, i) => {
          const t = ((frame * 0.02 + i / 5) % 1);
          return <line key={i} x1={420 + t * 380} y1={400 + t * 220} x2={440 + t * 380} y2={412 + t * 220} stroke="#fcd34d" strokeWidth={6} opacity={0.8 * (1 - t)} strokeLinecap="round" />;
        })}
        <rect x={x0 - 60} y={y} width={x1 - x0 + 120} height={40} rx={8} fill="#c8a06a" />
        <rect x={x0 - 30} y={y - 30} width={50} height={36} rx={6} fill="#6b7280" />
        <rect x={x1 - 20} y={y - 30} width={50} height={36} rx={6} fill="#6b7280" />
        <path d={`M ${x0} ${y - 12} Q ${(x0 + x1) / 2} ${y - 12 - h * 2} ${x1} ${y - 12}`} stroke={strip} strokeWidth={22} fill="none" strokeLinecap="round" />
        {snapped && burst < 0.98 && Array.from({ length: 8 }).map((_, i) => {
          const a = (i / 8) * Math.PI * 2, r1 = 60 + 120 * burst, r2 = r1 + 50;
          return <line key={i} x1={960 + Math.cos(a) * r1} y1={y + 60 + Math.sin(a) * r1} x2={960 + Math.cos(a) * r2} y2={y + 60 + Math.sin(a) * r2}
            stroke={C.accent} strokeWidth={8} strokeLinecap="round" opacity={1 - burst} />;
        })}
        <g transform="translate(1580 330)">
          <text x={40} y={-24} fill={C.dim} fontSize={30} fontWeight={700} textAnchor="middle">Stored energy</text>
          <rect x={0} y={0} width={80} height={420} rx={14} fill={C.panel2} />
          <rect x={6} y={6 + 408 * (1 - meter)} width={68} height={408 * meter} rx={10} fill={meter > 0.85 ? C.bad : C.accent} />
        </g>
      </svg>
      {snapped && (
        <div style={{ position: "absolute", left: 0, right: 0, top: 330, textAlign: "center", fontSize: 150, fontWeight: 900, color: C.accent,
          transform: `scale(${0.6 + 0.5 * burst}) rotate(-4deg)`, opacity: interpolate(frame - snapAt, [0, 4, fps * 1.5, fps * 2], [0, 1, 1, 0], clamp) }}>SNAP!</div>
      )}
      <div style={{ position: "absolute", left: 120, right: 120, top: 190, display: "flex", justifyContent: "center", gap: 24 }}>
        {labels.map((l, i) => (
          <div key={i} style={{ fontSize: 34, fontWeight: 800, padding: "12px 22px", borderRadius: 14, opacity: i <= phase ? 1 : 0.3,
            background: i === phase ? C.accent : C.panel, color: i === phase ? "#111" : C.ink }}>{i + 1}. {l}</div>
        ))}
      </div>
    </AbsoluteFill>
  );
};
