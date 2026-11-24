import React from "react";
import { AbsoluteFill, Img, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { C, Media, STAGE } from "../theme";
import { current } from "../design";
import { Tag } from "./chrome";
import { Cue, DP, Icon, MONO, SUB_PX, boxStyle, calm, cuer, dimWash, ease, hi, line, newest, onHi, pad2, pop, settle, tint } from "./kit";

// Cards: chapter title, paper figure, headline number, prediction prompt, quote. (Footage: media.tsx.)

const SHADOW = () => (current.light ? "0 18px 50px rgba(31,32,35,0.18), 0 0 0 1px rgba(31,32,35,0.08)" : "0 30px 80px rgba(0,0,0,0.55)");

// ---------- chapter ----------

// Number, title and the chapter's question, big and still. Defaults come from the segment (chapter, headline, question).
export const ChapterCard: React.FC<DP> = (p) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const { v, seg, chapters, born } = p;
  const num: number | undefined = v.number ?? seg.chapter;
  const title: string = v.title ?? seg.headline;
  const q: string | undefined = v.question ?? seg.question;
  const a = calm(frame, born, fps), b = calm(frame, born + 5, fps), c = calm(frame, settle(p, cuer(p, fps)(v.questionCue, p.pre + Math.round(fps * 0.7))), fps);
  return (
    <AbsoluteFill style={{ justifyContent: "center", padding: "0 200px 40px" }}>
      {num != null && (
        <div style={{ display: "flex", alignItems: "baseline", gap: 24, opacity: a, transform: `translateY(${(1 - a) * 20}px)` }}>
          <span style={{ fontFamily: MONO, fontSize: 130, fontWeight: 600, color: hi(), letterSpacing: -6, lineHeight: 1 }}>{pad2(num)}</span>
          {chapters > 1 && <span style={{ fontFamily: MONO, fontSize: 36, color: C.dim }}>/ {pad2(chapters)}</span>}
          <div style={{ flex: 1, height: 3, background: line(), alignSelf: "center", marginLeft: 20 }} />
        </div>
      )}
      <div style={{ fontSize: 96, fontWeight: 900, letterSpacing: -2.5, lineHeight: 1.04, marginTop: 36, maxWidth: 1450,
        opacity: b, transform: `translateY(${(1 - b) * 20}px)` }}>{title}</div>
      {q && <div style={{ fontSize: 52, fontWeight: 600, color: C.dim, marginTop: 36, maxWidth: 1450, lineHeight: 1.2, opacity: c }}>{q}</div>}
    </AbsoluteFill>
  );
};

// ---------- figure ----------

type Region = { x: number; y: number; w: number; h: number; cue?: Cue; label?: string };

// A paper figure shown whole, then one region at a time: on its cue the rest dims and the region is framed, its label on
// the frame. zoom: true cuts straight in on each region (no pan; text holds still).
export const Figure: React.FC<DP> = (p) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const v = p.v, c = cuer(p, fps);
  const m: Media = v.src;
  const regions: Region[] = v.regions ?? [];
  const fit = Math.min(STAGE.w / m.width, STAGE.h / m.height);
  const cw = m.width * fit, ch = m.height * fit, cx = STAGE.x + (STAGE.w - cw) / 2, cy = STAGE.y + (STAGE.h - ch) / 2;
  const at = regions.map((r, i) => c(r.cue, p.pre + Math.round(p.len * (0.25 + (0.6 * i) / Math.max(1, regions.length)))));
  const k = newest(at, frame);
  const prev = k < 0 ? -1 : newest(at.map((a, i) => (i === k ? Infinity : a)), frame);
  let s = 1, tx = 0, ty = 0;
  if (v.zoom && k >= 0) {
    const r = regions[k];
    s = Math.max(1, Math.min(2.2, (cw * 0.86) / (r.w * fit), (ch * 0.86) / (r.h * fit)));
    tx = Math.min(0, Math.max(cw - cw * s, cw / 2 - (r.x + r.w / 2) * fit * s));
    ty = Math.min(0, Math.max(ch - ch * s, ch / 2 - (r.y + r.h / 2) * fit * s));
  }
  const place = (r: Region) => ({ x: tx + r.x * fit * s, y: ty + r.y * fit * s, w: r.w * fit * s, h: r.h * fit * s });
  const enter = calm(frame, p.born, fps);
  // crossfade from the previous region to the new one (a zoom cuts, so its frame cuts too)
  const op = (i: number) => (i === k ? (v.zoom ? 1 : ease(frame, at[k], 7)) : i === prev && !v.zoom ? 1 - ease(frame, at[k], 7) : 0);
  return (
    <>
      <div style={{ position: "absolute", left: cx, top: cy, width: cw, height: ch, borderRadius: 14, overflow: "hidden", background: "#fff",
        boxShadow: SHADOW(), opacity: enter, transform: `translateY(${(1 - enter) * 40}px)` }}>
        <Img src={staticFile(m.src)} style={{ position: "absolute", left: tx, top: ty, width: cw * s, height: ch * s }} />
        {regions.map((r, i) => {
          const o = op(i);
          if (o <= 0) return null;
          const q = place(r);
          return <div key={i} style={{ position: "absolute", left: q.x - 4, top: q.y - 4, width: q.w + 8, height: q.h + 8, borderRadius: 8,
            border: `4px solid ${hi()}`, boxShadow: `0 0 0 4000px ${dimWash()}`, opacity: o }} />;
        })}
        {v.stamp && <Tag text={v.stamp} />}
      </div>
      {regions.map((r, i) => {
        const o = op(i);
        if (o <= 0 || !r.label) return null;
        const q = place(r);
        // a tab on the frame: above it if there's room under the header, else below, else inside its top-left corner
        const above = cy + q.y - 54 >= 150, below = cy + q.y + q.h + 58 <= STAGE.y + STAGE.h + 10;
        const top = above ? cy + q.y - 4 - 50 : below ? cy + q.y + q.h + 4 : cy + q.y + 10;
        const radius = above ? "8px 8px 0 0" : below ? "0 0 8px 8px" : "6px";
        const lw = r.label.length * 17 + 36; // rough width at 31px, to keep the tab on screen
        return <div key={i} style={{ position: "absolute", left: Math.min(cx + q.x - 4, 1920 - 64 - lw), top, height: 50, boxSizing: "border-box",
          display: "flex", alignItems: "center", padding: "0 18px", borderRadius: radius, background: hi(), color: onHi(), fontSize: 31,
          fontWeight: 800, whiteSpace: "nowrap", opacity: o }}>{r.label}</div>;
      })}
    </>
  );
};

// ---------- number ----------

// "0 of 15" draws its denominator: 15 squares, 0 filled.
const ofRe = /^\s*(\d+)\s*(?:of|\/)\s*(\d+)\s*$/i;
const Value: React.FC<{ text: string; size: number; color: string }> = ({ text, size, color }) => {
  const m = text.match(ofRe);
  const big = { fontSize: size, fontWeight: 900, letterSpacing: -size * 0.03, lineHeight: 1, color, fontVariantNumeric: "tabular-nums" as const };
  return m ? (
    <div style={big}>{m[1]}<span style={{ fontSize: size * 0.36, color: C.dim, fontWeight: 700, letterSpacing: 0, margin: `0 ${size * 0.12}px` }}>of</span>{m[2]}</div>
  ) : <div style={big}>{text}</div>;
};
const Pips: React.FC<{ text: string; color: string }> = ({ text, color }) => {
  const m = text.match(ofRe);
  if (!m || +m[2] > 60) return null;
  const k = +m[1], n = +m[2], per = n > 20 ? Math.ceil(n / 2) : n;
  return (
    <div style={{ display: "flex", flexWrap: "wrap", gap: 9, justifyContent: "center", width: per * 35 }}>
      {Array.from({ length: n }, (_, i) => (
        <div key={i} style={{ width: 26, height: 26, borderRadius: 5, boxSizing: "border-box", background: i < k ? color : "transparent",
          border: `3px solid ${i < k ? color : current.light ? "rgba(31,32,35,0.32)" : "rgba(255,255,255,0.3)"}` }} />
      ))}
    </div>
  );
};

// A headline number with its denominator and, on its cue, the comparison; the newer one carries the highlight. A label
// never shows without its number (alone, "...then crossed the real course" reads as the opposite claim): each column,
// number and label together, lands on its number's word.
export const NumberCard: React.FC<DP> = (p) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const v = p.v, c = cuer(p, fps);
  const cmp = v.compare as { value: string; label: string; cue?: Cue } | undefined;
  const a0 = settle(p, c(v.cue, p.pre));
  const a1 = cmp ? settle(p, c(cmp.cue, Math.max(a0 + fps, p.pre + Math.round(p.len * 0.5)))) : Infinity;
  const noteAt = settle(p, c(v.noteCue, (cmp ? a1 : a0) + Math.round(fps * 0.6)));
  const second = frame >= a1;
  const size = cmp ? 190 : 230;
  const col = (value: string, label: string, at: number, lit: boolean) => {
    const s = frame >= at ? pop(frame, at, fps) : 0;
    const color = lit ? hi() : C.ink;
    return (
      <div style={{ width: cmp ? 720 : 1200, display: "flex", flexDirection: "column", alignItems: "center", gap: 26, textAlign: "center",
        opacity: Math.min(1, s), transform: `scale(${0.85 + 0.15 * s})` }}>
        <Value text={value} size={size} color={color} />
        {v.pips !== false && <Pips text={value} color={color} />}
        <div style={{ fontSize: 42, fontWeight: 700, lineHeight: 1.2, maxWidth: cmp ? 640 : 1000 }}>{label}</div>
      </div>
    );
  };
  return (
    <AbsoluteFill style={{ justifyContent: "center", alignItems: "center", paddingBottom: 30 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 60 }}>
        {col(v.value, v.label, a0, !second)}
        {cmp && <div style={{ width: 3, height: 300, background: line(), opacity: frame >= a1 ? 1 : 0 }} />}
        {cmp && col(cmp.value, cmp.label, a1, second)}
      </div>
      {v.note && <div style={{ marginTop: 56, fontSize: 36, fontWeight: 600, color: C.ink, opacity: 0.8 * ease(frame, noteAt, 10),
        maxWidth: 1400, textAlign: "center" }}>{v.note}</div>}
    </AbsoluteFill>
  );
};

// ---------- predict ----------

// The question and its options held for a beat (the viewer guesses), then on revealCue the answer is marked and the rest
// dim. No timer, no sound: calm, not a quiz show.
export const Predict: React.FC<DP> = (p) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const v = p.v, c = cuer(p, fps);
  const opts: { label: string; sub?: string; cue?: Cue }[] = v.options ?? [];
  const oAt = opts.map((o, i) => settle(p, c(o.cue, p.pre + Math.round(fps * 0.35) + i * 6)));
  const rAt = c(v.revealCue, p.pre + Math.round(p.len * 0.8));
  const r = v.answer != null ? ease(frame, rAt, 8) : 0;
  const q = calm(frame, p.born, fps);
  const w = Math.min(500, (1560 - 40 * (opts.length - 1)) / Math.max(1, opts.length));
  return (
    <AbsoluteFill style={{ justifyContent: "center", alignItems: "center", paddingBottom: 30 }}>
      <div style={{ fontFamily: MONO, fontSize: 28, fontWeight: 600, color: hi(), letterSpacing: 1, opacity: q }}>{`// ${v.tag ?? "your guess"}`}</div>
      <div style={{ fontSize: 64, fontWeight: 800, letterSpacing: -1.2, lineHeight: 1.12, textAlign: "center", maxWidth: 1460, marginTop: 22,
        opacity: q, transform: `translateY(${(1 - q) * 16}px)` }}>{v.question}</div>
      <div style={{ display: "flex", gap: 40, marginTop: 64 }}>
        {opts.map((o, i) => {
          const s = frame >= oAt[i] ? pop(frame, oAt[i], fps) : 0;
          const right = i === v.answer; // the others dim but stay readable: they were the viewer's options
          return (
            <div key={i} style={{ position: "relative", width: w, minHeight: 150, ...boxStyle(right && r > 0), display: "flex", alignItems: "center",
              gap: 22, padding: "22px 28px", boxSizing: "border-box", opacity: Math.min(1, s) * (right ? 1 : 1 - 0.58 * r),
              transform: `scale(${0.9 + 0.1 * s})` }}>
              <div style={{ width: 54, height: 54, borderRadius: 27, flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center",
                fontFamily: MONO, fontSize: 28, fontWeight: 700, border: `3px solid ${right && r > 0 ? hi() : line()}`,
                color: right && r > 0 ? hi() : C.dim }}>{String.fromCharCode(65 + i)}</div>
              <div>
                <div style={{ fontSize: 40, fontWeight: 800, lineHeight: 1.1, letterSpacing: -0.5 }}>{o.label}</div>
                {o.sub && <div style={{ fontSize: SUB_PX, fontWeight: 600, color: C.dim, marginTop: 6 }}>{o.sub}</div>}
              </div>
              {right && r > 0 && (
                <div style={{ position: "absolute", right: -18, top: -18, width: 48, height: 48, borderRadius: 24, background: hi(),
                  display: "flex", alignItems: "center", justifyContent: "center", opacity: r, transform: `scale(${0.6 + 0.4 * r})` }}>
                  <Icon name="check" size={32} color={onHi()} />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </AbsoluteFill>
  );
};

// ---------- quote ----------

// A sourced quote (<= 25 words), text only: no portrait, no logo. "mark" underlines one phrase of it on its cue.
export const Quote: React.FC<DP> = (p) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const v = p.v;
  const a = calm(frame, p.born, fps), b = calm(frame, p.born + 8, fps);
  const mp = ease(frame, cuer(p, fps)(v.markCue, p.pre + Math.round(fps * 1.2)), 14);
  const text: string = v.text, mark: string | undefined = v.mark;
  const j = mark ? text.indexOf(mark) : -1;
  const body = j < 0 ? text : (
    <>
      {text.slice(0, j)}
      <span style={{ backgroundImage: `linear-gradient(${tint(0.3)}, ${tint(0.3)})`, backgroundRepeat: "no-repeat",
        backgroundPosition: "0 88%", backgroundSize: `${mp * 100}% 38%`, WebkitBoxDecorationBreak: "clone", boxDecorationBreak: "clone" }}>{mark}</span>
      {text.slice(j + mark!.length)}
    </>
  );
  return (
    <AbsoluteFill style={{ justifyContent: "center", padding: "0 250px 50px" }}>
      <div style={{ position: "relative", opacity: a, transform: `translateY(${(1 - a) * 16}px)` }}>
        <div style={{ position: "absolute", left: -118, top: -46, fontSize: 230, fontWeight: 900, color: hi(), lineHeight: 1,
          fontFamily: "Georgia, 'Times New Roman', serif" }}>“</div>
        <div style={{ fontSize: 62, fontWeight: 700, lineHeight: 1.24, letterSpacing: -1 }}>{body}</div>
      </div>
      <div style={{ marginTop: 46, display: "flex", alignItems: "baseline", gap: 18, flexWrap: "wrap", opacity: b }}>
        <div style={{ width: 46, height: 3, background: hi(), alignSelf: "center" }} />
        <span style={{ fontSize: 38, fontWeight: 800 }}>{v.name}</span>
        {v.role && <span style={{ fontSize: 32, fontWeight: 600, color: C.dim }}>{v.role}</span>}
      </div>
      {v.source && <div style={{ marginTop: 12, marginLeft: 64, fontSize: 28, color: C.dim, fontFamily: MONO, opacity: b }}>{v.source}</div>}
    </AbsoluteFill>
  );
};
