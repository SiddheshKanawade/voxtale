import React from "react";
import { AbsoluteFill, Easing, Img, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { C, Media, STAGE, Visual, Word } from "./theme";
import { Lockup } from "./brand";
import { Kitting, Snap } from "./diagrams";

type P = { v: Visual; dur: number; index: number; words: Word[] };

// Frame at which a cue word is spoken in this beat (first word that starts with the cue), or a fallback.
export const cueFrame = (words: Word[], cue: string | undefined, fps: number, fallback: number) => {
  if (!cue) return fallback;
  const n = (s: string) => s.toLowerCase().replace(/[^a-z0-9]/g, "");
  const w = words.find((w) => n(w.text).startsWith(n(cue)));
  return w ? Math.round(w.start * fps) : fallback;
};

// Reveal times for a list of items: each item's `cue` word if present, else spread over the first 60% of the beat.
const reveals = (items: { cue?: string }[], words: Word[], fps: number, dur: number) =>
  items.map((it, i) => cueFrame(words, it.cue, fps, Math.round((i / Math.max(1, items.length)) * dur * 0.6)));

const pop = (frame: number, at: number, fps: number) => spring({ frame: frame - at, fps, config: { damping: 14, mass: 0.6 } });

// A static image should not hold longer than this; past it, cut to a tighter framing.
export const HOLD_CAP_S = 7;

// Frame of the punch-in cut for a long static beat: the clause break (word ending in punctuation) nearest
// the middle, so the cut lands on a natural pause in the narration. Infinity when no cut is needed.
const punchCut = (words: Word[], dur: number, fps: number) => {
  if (dur <= HOLD_CAP_S * fps) return Infinity;
  const breaks = words.filter((w) => /[.,?!:;]$/.test(w.text)).map((w) => Math.round(w.end * fps));
  const mid = dur / 2;
  const best = breaks.filter((b) => b > fps * 2 && b < dur - fps * 2).sort((a, b) => Math.abs(a - mid) - Math.abs(b - mid))[0];
  return best ?? Math.round(mid);
};

export const Scene: React.FC<P> = (p) => {
  const frame = useCurrentFrame();
  const o = interpolate(frame, [0, 6, p.dur - 5, p.dur], [0, 1, 1, 0], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
  const Comp = SCENES[p.v.type] ?? Missing;
  return <AbsoluteFill style={{ opacity: o }}><Comp {...p} /></AbsoluteFill>;
};

const Missing: React.FC<P> = ({ v }) => <AbsoluteFill style={{ justifyContent: "center", alignItems: "center", fontSize: 60, color: C.bad }}>Unknown scene: {v.type}</AbsoluteFill>;

// ---------- media scenes ----------

const Card: React.FC<{ m: Media; box: { x: number; y: number; w: number; h: number }; style?: React.CSSProperties; imgStyle?: React.CSSProperties; bg?: string }> = ({ m, box, style, imgStyle, bg = "#fff" }) => {
  const fit = Math.min(box.w / m.width, box.h / m.height);
  const w = m.width * fit, h = m.height * fit;
  return (
    <div style={{ position: "absolute", left: box.x + (box.w - w) / 2, top: box.y + (box.h - h) / 2, width: w, height: h,
      borderRadius: 14, overflow: "hidden", boxShadow: "0 30px 80px rgba(0,0,0,0.55)", background: bg, ...style }}>
      <Img src={staticFile(m.src)} style={{ width: "100%", height: "100%", objectFit: "contain", ...imgStyle }} />
    </div>
  );
};

// Page screenshot; pushes in on the highlighted sentence when there is one.
const Shot: React.FC<P> = ({ v, dur, words }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const m = v.src!;
  const punch = frame >= punchCut(words, dur, fps) ? 1.3 : 1;
  const enter = spring({ frame, fps, config: { damping: 18, mass: 0.6 } });
  const fit = Math.min(STAGE.w / m.width, STAGE.h / m.height);
  const cw = m.width * fit, ch = m.height * fit;
  // Zoom so the highlight fills ~90% of the card width, panned to the card centre, clamped so no blank edge shows.
  let tf = `scale(${interpolate(frame, [0, dur], [1, 1.08]) * punch})`, origin = "50% 30%";
  if (m.highlight) {
    const t = interpolate(frame, [fps * 0.4, fps * 2], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.inOut(Easing.cubic) });
    const target = Math.max(1, Math.min(2.2, (cw * 0.9) / (m.highlight.w * fit))) * punch;
    const s = interpolate(t, [0, 1], [1, target]) * interpolate(frame, [0, dur], [1, 1.03]);
    const hx = (m.highlight.x + m.highlight.w / 2) * fit, hy = (m.highlight.y + m.highlight.h / 2) * fit;
    const tx = Math.min(0, Math.max(cw - cw * s, (cw / 2 - hx * s) * t));
    const ty = Math.min(0, Math.max(ch - ch * s, (ch / 2 - hy * s) * t));
    tf = `translate(${tx}px, ${ty}px) scale(${s})`;
    origin = "0 0";
  }
  const st = pop(frame, Math.round(dur * 0.35), fps);
  return (
    <>
      <Card m={m} box={STAGE} style={{ transform: `translateY(${(1 - enter) * 60}px)` }}
        imgStyle={{ transform: tf, transformOrigin: origin }} />
      {v.stamp && (
        <div style={{ position: "absolute", right: 200, top: 230, transform: `rotate(-8deg) scale(${0.6 + st * 0.4})`, opacity: st,
          border: `6px solid ${C.bad}`, color: C.bad, fontWeight: 900, fontSize: 64, padding: "8px 28px", borderRadius: 12,
          letterSpacing: 4, textTransform: "uppercase", background: "rgba(15,15,17,0.85)" }}>{v.stamp}</div>
      )}
    </>
  );
};

// Photo: full-bleed Ken Burns (cover) or a card (contain), with optional overlay, badge and location tags.
const Photo: React.FC<P> = ({ v, dur, index, words }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const m = v.src!;
  const pan = v.pan ?? (["in", "out", "left", "right"] as const)[index % 4];
  const t = frame / dur;
  const s = pan === "out" ? 1.18 - 0.12 * t : 1.06 + 0.12 * t;
  const x = pan === "left" ? 3 - 6 * t : pan === "right" ? -3 + 6 * t : 0;
  // Past the hold cap, cut to a tighter frame off-centre (a "punch-in"), so the shot reads as new.
  const punched = frame >= punchCut(words, dur, fps);
  const pk = punched ? 1.35 : 1, po = punched ? "62% 40%" : "50% 50%";
  const media = v.fit === "contain"
    ? <Card m={m} box={STAGE} imgStyle={{ transform: `scale(${(1 + 0.06 * t) * pk})`, transformOrigin: po }} />
    : (
      <AbsoluteFill>
        <Img src={staticFile(m.src)} style={{ width: "100%", height: "100%", objectFit: "cover", transform: `scale(${s * pk}) translateX(${x}%)`, transformOrigin: po }} />
        <AbsoluteFill style={{ background: `linear-gradient(rgba(${C.bgRgb},0.85) 0%, rgba(${C.bgRgb},0) 28%, rgba(${C.bgRgb},0) 62%, rgba(${C.bgRgb},0.9) 100%)` }} />
      </AbsoluteFill>
    );
  const ov = pop(frame, 6, fps);
  const bd = pop(frame, cueFrame(words, (v as any).badgeCue, fps, Math.round(fps * 0.8)), fps);
  return (
    <>
      {media}
      {v.overlay && (
        <AbsoluteFill style={{ justifyContent: "center", alignItems: "center" }}>
          <div style={{ fontSize: 96, fontWeight: 900, letterSpacing: -2, textAlign: "center", maxWidth: 1400, lineHeight: 1.05,
            color: "#f5f4f2", opacity: ov, transform: `translateY(${(1 - ov) * 40}px)`, textShadow: "0 6px 30px rgba(0,0,0,0.85)" }}>{v.overlay}</div>
        </AbsoluteFill>
      )}
      {v.badge && (
        <div style={{ position: "absolute", right: 90, top: 200, background: C.accent, color: "#111", fontWeight: 900, fontSize: 52,
          padding: "14px 28px", borderRadius: 14, transform: `scale(${bd}) rotate(${(1 - bd) * 10}deg)`, boxShadow: "0 20px 50px rgba(0,0,0,0.5)" }}>{v.badge}</div>
      )}
      {v.tags?.map((tag, i) => {
        const tp = pop(frame, Math.round(fps * (0.8 + i * 0.7)), fps);
        return (
          <div key={tag} style={{ position: "absolute", left: 90, top: 560 + i * 90, background: "rgba(15,15,17,0.88)", border: `3px solid ${C.accent}`,
            borderRadius: 40, padding: "10px 24px", fontSize: 36, fontWeight: 800, color: "#f5f4f2", opacity: tp, transform: `translateX(${(1 - tp) * -60}px)` }}>📍 {tag}</div>
        );
      })}
    </>
  );
};

// Two images side by side, each sliding in when its label's cue word is spoken.
const Split: React.FC<P> = ({ v, dur, words }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const items = v.items!;
  const at = reveals(items, words, fps, dur);
  const w = (STAGE.w - 40) / 2;
  return (
    <>
      {items.map((it, i) => {
        const s = pop(frame, at[i], fps);
        return (
          <div key={i} style={{ opacity: s, transform: `translateX(${(1 - s) * (i ? 120 : -120)}px)` }}>
            <Card m={it} box={{ x: STAGE.x + i * (w + 40), y: STAGE.y, w, h: STAGE.h - 70 }} bg="#000"
              imgStyle={{ objectFit: "cover", transform: `scale(${1.05 + 0.05 * (frame / dur)})` }} />
            <div style={{ position: "absolute", left: STAGE.x + i * (w + 40), width: w, top: STAGE.y + STAGE.h - 50, textAlign: "center",
              fontSize: 38, fontWeight: 800 }}>{it.label}</div>
          </div>
        );
      })}
    </>
  );
};

// A row of images revealed in turn (on cue words); the newest one is enlarged and outlined.
const Panels: React.FC<P> = ({ v, dur, words }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const items = v.items!;
  const at = reveals(items, words, fps, dur);
  const gap = 28, w = (STAGE.w - gap * (items.length - 1)) / items.length;
  const current = at.reduce((c, a, i) => (frame >= a ? i : c), 0);
  return (
    <>
      {items.map((it, i) => {
        const s = pop(frame, at[i], fps);
        const on = i === current;
        return (
          <div key={i} style={{ opacity: s, transform: `translateY(${(1 - s) * 80}px)` }}>
            <Card m={it} box={{ x: STAGE.x + i * (w + gap), y: STAGE.y + 10, w, h: STAGE.h - 90 }} bg="#000"
              style={{ transform: `scale(${on ? 1.04 : 0.96})`, outline: on ? `5px solid ${C.accent}` : "none", filter: on ? "none" : "brightness(0.7)" }} />
            <div style={{ position: "absolute", left: STAGE.x + i * (w + gap), width: w, top: STAGE.y + STAGE.h - 60, textAlign: "center",
              fontSize: 40, fontWeight: 900, color: on ? C.accent : C.ink }}>{it.label}</div>
          </div>
        );
      })}
    </>
  );
};

// ---------- typographic / diagram scenes ----------

const Kinetic: React.FC<P> = ({ v }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const parts = v.text!.split(" ");
  return (
    <AbsoluteFill style={{ justifyContent: "center", alignItems: "center", flexDirection: "column" }}>
      <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "center", maxWidth: 1500, gap: "0 28px" }}>
        {parts.map((p, i) => {
          const s = pop(frame, i * 4, fps);
          return <span key={i} style={{ fontSize: 130, fontWeight: 900, letterSpacing: -3, display: "inline-block",
            opacity: s, transform: `translateY(${(1 - s) * 50}px) scale(${0.8 + 0.2 * s})`, color: p === "→" ? C.accent : C.ink }}>{p}</span>;
        })}
      </div>
      {v.sub && <div style={{ marginTop: 30, fontSize: 40, color: C.dim, fontWeight: 600, opacity: pop(frame, parts.length * 4 + 4, fps) }}>{v.sub}</div>}
    </AbsoluteFill>
  );
};

const Acronym: React.FC<P> = ({ v, dur, words }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const letters = v.letters!;
  const at = reveals(letters.map((l: any) => ({ cue: l.cue ?? l.word })), words, fps, dur);
  return (
    <AbsoluteFill style={{ justifyContent: "center", alignItems: "center", flexDirection: "column", gap: 10 }}>
      {letters.map((l, i) => {
        const s = pop(frame, i * 5, fps);
        const w = interpolate(frame - at[i], [0, 10], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
        return (
          <div key={i} style={{ display: "flex", alignItems: "baseline", width: 900, opacity: s }}>
            <span style={{ fontSize: 170, fontWeight: 900, color: C.accent, width: 170, textAlign: "right", marginRight: 4, transform: `scale(${s})` }}>{l.letter}</span>
            <span style={{ fontSize: 96, fontWeight: 800, clipPath: `inset(0 ${(1 - w) * 100}% 0 0)` }}>{l.word.slice(1)}</span>
          </div>
        );
      })}
    </AbsoluteFill>
  );
};

const Flow: React.FC<P> = ({ v, dur, words }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const steps = v.steps!;
  const at = reveals(steps as any, words, fps, dur);
  const w = 420, gap = (STAGE.w - w * steps.length) / (steps.length - 1);
  const current = at.reduce((c, a, i) => (frame >= a ? i : c), 0);
  return (
    <>
      {steps.map((st, i) => {
        const s = pop(frame, at[i], fps);
        const x = STAGE.x + i * (w + gap);
        const on = i === current;
        const arrow = i < steps.length - 1 ? interpolate(frame - at[i + 1], [-12, 0], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" }) : 0;
        return (
          <React.Fragment key={i}>
            <div style={{ position: "absolute", left: x, top: STAGE.y + 90, width: w, height: 440, borderRadius: 28,
              background: on ? C.panel2 : C.panel, border: `4px solid ${on ? C.accent : "transparent"}`, display: "flex", flexDirection: "column",
              alignItems: "center", justifyContent: "center", gap: 18, opacity: s, transform: `translateY(${(1 - s) * 60}px) scale(${on ? 1.04 : 1})` }}>
              <div style={{ fontSize: 150, transform: `scale(${on ? 1 + 0.05 * Math.sin(frame / 6) : 1})` }}>{st.icon}</div>
              <div style={{ fontSize: 54, fontWeight: 900 }}>{st.label}</div>
              {st.sub && <div style={{ fontSize: 32, color: C.dim, fontWeight: 600, textAlign: "center", padding: "0 24px" }}>{st.sub}</div>}
            </div>
            {i < steps.length - 1 && (v as any).arrows !== false && (
              <div style={{ position: "absolute", left: x + w + 14, top: STAGE.y + 290, width: (gap - 28) * arrow, height: 10,
                background: C.accent, borderRadius: 5 }}>
                <div style={{ position: "absolute", right: -6, top: -13, width: 0, height: 0, borderLeft: `26px solid ${C.accent}`,
                  borderTop: "18px solid transparent", borderBottom: "18px solid transparent", opacity: arrow > 0.9 ? 1 : 0 }} />
              </div>
            )}
          </React.Fragment>
        );
      })}
    </>
  );
};

const Hub: React.FC<P> = ({ v, dur, words }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const spokes = v.spokes!;
  const at = reveals(spokes as any, words, fps, dur);
  // Centre node on the left, spokes stacked on the right: reads left-to-right as "one brain -> many bodies".
  const cx = 600, cy = 520;
  const c = pop(frame, 0, fps);
  const pos = spokes.map((_, i) => ({ x: 1380, y: cy + (i - (spokes.length - 1) / 2) * 235 }));
  const current = at.reduce((k, a, i) => (frame >= a ? i : k), -1);
  return (
    <>
      <svg style={{ position: "absolute", inset: 0 }} width={1920} height={1080}>
        {spokes.map((_, i) => {
          const p = interpolate(frame - at[i], [0, 12], [0, 1], { extrapolateLeft: "clamp", extrapolateRight: "clamp" });
          const x1 = cx + 150, x2 = x1 + (pos[i].x - 190 - x1) * p, y2 = cy + (pos[i].y - cy) * p;
          return <line key={i} x1={x1} y1={cy} x2={x2} y2={y2} stroke={C.accent} strokeWidth={6} strokeDasharray="14 10" strokeDashoffset={-frame} />;
        })}
      </svg>
      <div style={{ position: "absolute", left: cx - 150, top: cy - 150, width: 300, height: 300, borderRadius: 150, background: C.panel2,
        border: `6px solid ${C.accent}`, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
        transform: `scale(${c * (1 + 0.03 * Math.sin(frame / 8))})`, boxShadow: `0 0 ${60 + 20 * Math.sin(frame / 8)}px rgba(240,201,69,0.35)` }}>
        <div style={{ fontSize: 120 }}>{v.center!.icon}</div>
        <div style={{ fontSize: 28, fontWeight: 800, textAlign: "center" }}>{v.center!.label}</div>
      </div>
      {spokes.map((sp, i) => {
        const s = pop(frame, at[i] + 8, fps);
        const on = i === current;
        return (
          <div key={i} style={{ position: "absolute", left: pos[i].x - 190, top: pos[i].y - 100, width: 520, height: 200, borderRadius: 24,
            background: on ? C.panel2 : C.panel, border: `4px solid ${on ? C.accent : "transparent"}`, display: "flex", alignItems: "center",
            gap: 26, padding: "0 30px", opacity: s, transform: `translateX(${(1 - s) * 80}px)` }}>
            <div style={{ fontSize: 100 }}>{sp.icon}</div>
            <div>
              <div style={{ fontSize: 40, fontWeight: 800 }}>{sp.label}</div>
              {sp.sub && <div style={{ fontSize: 28, color: C.dim, fontWeight: 600 }}>{sp.sub}</div>}
            </div>
          </div>
        );
      })}
    </>
  );
};

const Stat: React.FC<P> = ({ v, dur, words }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const items = v.items!;
  const at = reveals(items, words, fps, dur);
  const width = v.side ? STAGE.w * 0.55 : STAGE.w;
  return (
    <>
      <div style={{ position: "absolute", left: STAGE.x, top: STAGE.y, width, height: STAGE.h, display: "flex", flexDirection: v.side ? "column" : "row",
        justifyContent: "center", alignItems: "center", gap: 80 }}>
        {items.map((it, i) => {
          const s = pop(frame, at[i], fps);
          const n = interpolate(frame - at[i], [0, fps * 1.4], [it.from ?? 0, it.to], { extrapolateLeft: "clamp", extrapolateRight: "clamp", easing: Easing.out(Easing.cubic) });
          return (
            <div key={i} style={{ textAlign: "center", opacity: s, transform: `scale(${0.7 + 0.3 * s})` }}>
              <div style={{ fontSize: v.side ? 240 : 170, fontWeight: 900, color: C.accent, letterSpacing: -6, fontVariantNumeric: "tabular-nums" }}>
                {it.prefix ?? ""}{Math.round(n)}{it.suffix ?? ""}
              </div>
              <div style={{ fontSize: 44, fontWeight: 700, color: C.ink }}>{it.label}</div>
            </div>
          );
        })}
      </div>
      {v.side && (() => {
        const s = pop(frame, 10, fps);
        return <Card m={v.side} box={{ x: STAGE.x + STAGE.w * 0.58, y: STAGE.y - 20, w: STAGE.w * 0.42, h: STAGE.h + 20 }} bg={C.panel}
          style={{ opacity: s, transform: `translateX(${(1 - s) * 200}px) rotate(${(1 - s) * 4}deg)` }} />;
      })()}
    </>
  );
};

const Versus: React.FC<P> = ({ v, dur, words }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const [aL, aR] = reveals([v.left as any, v.right as any], words, fps, dur);
  const box = (side: { icon: string; title: string; sub: string }, s: number, x: number, hot: boolean) => (
    <div style={{ position: "absolute", left: x, top: STAGE.y + 60, width: 700, height: 520, borderRadius: 30, background: hot ? C.panel2 : C.panel,
      border: `4px solid ${hot ? C.accent : "transparent"}`, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
      gap: 20, opacity: s, transform: `translateY(${(1 - s) * 80}px)` }}>
      <div style={{ fontSize: 170, transform: `rotate(${Math.sin(frame / 10) * 4}deg)` }}>{side.icon}</div>
      <div style={{ fontSize: 64, fontWeight: 900 }}>{side.title}</div>
      <div style={{ fontSize: 38, color: C.dim, fontWeight: 600, textAlign: "center", padding: "0 40px" }}>{side.sub}</div>
    </div>
  );
  const sL = pop(frame, aL, fps), sR = pop(frame, aR, fps);
  return (
    <>
      {box(v.left!, sL, 200, frame < aR)}
      <div style={{ position: "absolute", left: 960 - 60, top: STAGE.y + 260, width: 120, textAlign: "center", fontSize: 64, fontWeight: 900,
        color: C.accent, opacity: sR, transform: `scale(${sR})` }}>vs</div>
      {box(v.right!, sR, 1020, frame >= aR)}
    </>
  );
};

const ICON: Record<string, string> = { ok: "✅", no: "❌", warn: "⚠️" };
const Checklist: React.FC<P> = ({ v, dur, words }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const items = v.items!;
  const at = reveals(items, words, fps, dur);
  return (
    <AbsoluteFill style={{ justifyContent: "center", alignItems: "center" }}>
      <div style={{ width: 1200, marginTop: 40 }}>
        <div style={{ fontSize: 72, fontWeight: 900, marginBottom: 30, opacity: pop(frame, 0, fps) }}>{v.title}</div>
        {items.map((it, i) => {
          const s = pop(frame, at[i], fps);
          return (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 28, fontSize: 52, fontWeight: 700, padding: "18px 28px",
              marginBottom: 14, borderRadius: 18, background: C.panel, opacity: s, transform: `translateX(${(1 - s) * -80}px)` }}>
              <span style={{ fontSize: 60, transform: `scale(${s})`, display: "inline-block" }}>{ICON[it.status]}</span>{it.text}
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

const End: React.FC<P> = ({ v, words }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = pop(frame, 0, fps), b = pop(frame, 20, fps);
  const sub = pop(frame, cueFrame(words, (v as any).subscribeCue, fps, 20), fps);
  return (
    <AbsoluteFill style={{ justifyContent: "center", alignItems: "center", flexDirection: "column", gap: 40 }}>
      <div style={{ fontSize: 120, fontWeight: 900, letterSpacing: -3, opacity: s, transform: `scale(${0.8 + 0.2 * s})` }}>
        <Lockup size={110} />
      </div>
      <div style={{ fontSize: 44, color: C.dim, fontWeight: 600, opacity: b }}>Sources linked in the description</div>
      <div style={{ background: "#ff0033", color: "#fff", fontSize: 50, fontWeight: 900, padding: "18px 48px", borderRadius: 50,
        transform: `scale(${sub * (1 + 0.04 * Math.sin(frame / 5))})` }}>SUBSCRIBE</div>
    </AbsoluteFill>
  );
};

const KittingScene: React.FC<P> = ({ dur, words }) => <Kitting dur={dur} words={words} />;
const SnapScene: React.FC<P> = ({ dur, words }) => <Snap dur={dur} words={words} />;

const SCENES: Record<string, React.FC<P>> = {
  shot: Shot, photo: Photo, split: Split, panels: Panels, kinetic: Kinetic, acronym: Acronym, flow: Flow, hub: Hub,
  stat: Stat, versus: Versus, checklist: Checklist, end: End, kitting: KittingScene, snap: SnapScene,
};
