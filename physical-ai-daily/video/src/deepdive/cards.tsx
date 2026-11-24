import React from "react";
import { AbsoluteFill, Img, OffthreadVideo, Sequence, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { C, Media, STAGE } from "../theme";
import { current } from "../design";
import { Tag } from "./chrome";
import { Cue, DP, Icon, MONO, boxStyle, calm, cueAt, dimWash, ease, hi, line, newest, onHi, pad2, pop, revealAt, tint } from "./kit";

// Cards: chapter title, paper figure, headline number, prediction prompt, side-by-side pair, quote.

const SHADOW = () => (current.light ? "0 18px 50px rgba(31,32,35,0.18), 0 0 0 1px rgba(31,32,35,0.08)" : "0 30px 80px rgba(0,0,0,0.55)");

// ---------- chapter ----------

// Number, title and the chapter's question, big and still. Defaults come from the segment (chapter, headline, question).
export const ChapterCard: React.FC<DP> = ({ v, words, seg, chapters }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const num: number | undefined = v.number ?? seg.chapter;
  const title: string = v.title ?? seg.headline;
  const q: string | undefined = v.question ?? seg.question;
  const a = calm(frame, 0, fps), b = calm(frame, 5, fps), c = calm(frame, cueAt(words, v.questionCue, fps, Math.round(fps * 0.7)), fps);
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
export const Figure: React.FC<DP> = ({ v, dur, words }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const m: Media = v.src;
  const regions: Region[] = v.regions ?? [];
  const fit = Math.min(STAGE.w / m.width, STAGE.h / m.height);
  const cw = m.width * fit, ch = m.height * fit, cx = STAGE.x + (STAGE.w - cw) / 2, cy = STAGE.y + (STAGE.h - ch) / 2;
  const at = regions.map((r, i) => cueAt(words, r.cue, fps, Math.round(dur * (0.25 + (0.6 * i) / Math.max(1, regions.length)))));
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
  const enter = calm(frame, 0, fps);
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
          const p = place(r);
          return <div key={i} style={{ position: "absolute", left: p.x - 4, top: p.y - 4, width: p.w + 8, height: p.h + 8, borderRadius: 8,
            border: `4px solid ${hi()}`, boxShadow: `0 0 0 4000px ${dimWash()}`, opacity: o }} />;
        })}
        {v.stamp && <Tag text={v.stamp} />}
      </div>
      {regions.map((r, i) => {
        const o = op(i);
        if (o <= 0 || !r.label) return null;
        const p = place(r);
        // a tab on the frame: above it if there's room under the header, else below, else inside its top-left corner
        const above = cy + p.y - 50 >= 150, below = cy + p.y + p.h + 54 <= STAGE.y + STAGE.h + 10;
        const top = above ? cy + p.y - 4 - 46 : below ? cy + p.y + p.h + 4 : cy + p.y + 10;
        const radius = above ? "8px 8px 0 0" : below ? "0 0 8px 8px" : "6px";
        const lw = r.label.length * 15.5 + 32; // rough width at 27px, to keep the tab on screen
        return <div key={i} style={{ position: "absolute", left: Math.min(cx + p.x - 4, 1920 - 64 - lw), top, height: 46, boxSizing: "border-box",
          display: "flex", alignItems: "center", padding: "0 16px", borderRadius: radius, background: hi(), color: onHi(), fontSize: 27,
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

// A headline number with its denominator and, on its cue, the comparison; the newer one carries the highlight. The
// label holds the stage from the start and the number lands on its word (a number shown before it's said spoils it).
export const NumberCard: React.FC<DP> = ({ v, dur, words }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const cmp = v.compare as { value: string; label: string; cue?: Cue } | undefined;
  const a0 = cueAt(words, v.cue, fps, Math.round(0.4 * fps));
  const a1 = cmp ? cueAt(words, cmp.cue, fps, Math.max(a0 + fps, Math.round(dur * 0.5))) : Infinity;
  const noteAt = cueAt(words, v.noteCue, fps, (cmp ? a1 : a0) + Math.round(fps * 0.6));
  const second = frame >= a1;
  const size = cmp ? 190 : 230;
  const col = (value: string, label: string, labelAt: number, at: number, lit: boolean) => {
    const s = frame >= at ? pop(frame, at, fps) : 0, l = calm(frame, labelAt, fps);
    const color = lit ? hi() : C.ink;
    return (
      <div style={{ width: cmp ? 720 : 1200, display: "flex", flexDirection: "column", alignItems: "center", gap: 26, textAlign: "center" }}>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 26, opacity: s, transform: `scale(${0.85 + 0.15 * s})` }}>
          <Value text={value} size={size} color={color} />
          {v.pips !== false && <Pips text={value} color={color} />}
        </div>
        <div style={{ fontSize: 42, fontWeight: 700, lineHeight: 1.2, maxWidth: cmp ? 640 : 1000, opacity: l }}>{label}</div>
      </div>
    );
  };
  return (
    <AbsoluteFill style={{ justifyContent: "center", alignItems: "center", paddingBottom: 30 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 60 }}>
        {col(v.value, v.label, 0, a0, !second)}
        {cmp && <div style={{ width: 3, height: 300, background: line(), opacity: frame >= a1 ? 1 : 0 }} />}
        {cmp && col(cmp.value, cmp.label, a1, a1, second)}
      </div>
      {v.note && <div style={{ marginTop: 56, fontSize: 32, fontWeight: 600, color: C.dim, maxWidth: 1400, textAlign: "center",
        opacity: ease(frame, noteAt, 10) }}>{v.note}</div>}
    </AbsoluteFill>
  );
};

// ---------- predict ----------

// The question and its options held for a beat (the viewer guesses), then on revealCue the answer is marked and the rest
// dim. No timer, no sound: calm, not a quiz show.
export const Predict: React.FC<DP> = ({ v, dur, words }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const opts: { label: string; sub?: string; cue?: Cue }[] = v.options ?? [];
  const oAt = opts.map((o, i) => cueAt(words, o.cue, fps, Math.round(fps * 0.35) + i * 6));
  const rAt = cueAt(words, v.revealCue, fps, Math.round(dur * 0.8));
  const r = v.answer != null ? ease(frame, rAt, 8) : 0;
  const q = calm(frame, 0, fps);
  const w = Math.min(500, (1560 - 40 * (opts.length - 1)) / Math.max(1, opts.length));
  return (
    <AbsoluteFill style={{ justifyContent: "center", alignItems: "center", paddingBottom: 30 }}>
      <div style={{ fontFamily: MONO, fontSize: 26, fontWeight: 600, color: hi(), letterSpacing: 1, opacity: q }}>{`// ${v.tag ?? "your guess"}`}</div>
      <div style={{ fontSize: 64, fontWeight: 800, letterSpacing: -1.2, lineHeight: 1.12, textAlign: "center", maxWidth: 1460, marginTop: 22,
        opacity: q, transform: `translateY(${(1 - q) * 16}px)` }}>{v.question}</div>
      <div style={{ display: "flex", gap: 40, marginTop: 64 }}>
        {opts.map((o, i) => {
          const s = frame >= oAt[i] ? pop(frame, oAt[i], fps) : 0;
          const right = i === v.answer; // the others dim but stay readable: they were the viewer's options
          return (
            <div key={i} style={{ position: "relative", width: w, minHeight: 150, ...boxStyle(right && r > 0), display: "flex", alignItems: "center",
              gap: 22, padding: "22px 28px", boxSizing: "border-box", opacity: s * (right ? 1 : 1 - 0.58 * r),
              transform: `scale(${0.9 + 0.1 * s})` }}>
              <div style={{ width: 54, height: 54, borderRadius: 27, flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center",
                fontFamily: MONO, fontSize: 28, fontWeight: 700, border: `3px solid ${right && r > 0 ? hi() : line()}`,
                color: right && r > 0 ? hi() : C.dim }}>{String.fromCharCode(65 + i)}</div>
              <div>
                <div style={{ fontSize: 38, fontWeight: 800, lineHeight: 1.1, letterSpacing: -0.5 }}>{o.label}</div>
                {o.sub && <div style={{ fontSize: 24, fontWeight: 600, color: C.dim, marginTop: 6 }}>{o.sub}</div>}
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

// ---------- pair ----------

type PItem = Media & { label: string; stamp?: string; cue?: Cue };
const isClip = (src: string) => /\.(mp4|webm|mov|m4v)$/i.test(src);

// Two clips or images side by side, labelled on top ("Simulation" / "Real"). Clips are muted and hold their last frame.
// focus: [{item, cue}] lights one side and dims the other.
export const Pair: React.FC<DP> = ({ v, dur, words }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const items: PItem[] = (v.items ?? []).slice(0, 2);
  const at = revealAt(items, words, fps, dur);
  const focus: { item: number; cue?: Cue }[] = v.focus ?? [];
  const fAt = focus.map((f) => cueAt(words, f.cue, fps, Math.round(dur * 0.6)));
  const fk = newest(fAt, frame);
  const lit = fk >= 0 ? focus[fk].item : -1;
  const gap = 48, bw = (STAGE.w - gap) / 2, top = STAGE.y + 58, bh = STAGE.h - 58;
  return (
    <>
      {items.map((it, i) => {
        const fit = Math.min(bw / it.width, bh / it.height);
        const w = it.width * fit, h = it.height * fit, x = STAGE.x + i * (bw + gap) + (bw - w) / 2, y = top + (bh - h) / 2;
        const s = frame >= at[i] ? calm(frame, at[i], fps) : 0;
        const on = i === lit, dim = lit >= 0 && !on;
        const clipF = Math.max(1, Math.floor((it.duration ?? dur / fps) * fps));
        const media: React.CSSProperties = { width: "100%", height: "100%", objectFit: "contain", display: "block" };
        return (
          <div key={i} style={{ opacity: s * (dim ? 0.4 : 1), transform: `translateY(${(1 - s) * 30}px)` }}>
            <div style={{ position: "absolute", left: x, width: w, top: y - 56, textAlign: "center", fontSize: 36, fontWeight: 800,
              letterSpacing: -0.4, color: on ? hi() : C.ink }}>{it.label}</div>
            <div style={{ position: "absolute", left: x, top: y, width: w, height: h, borderRadius: 14, overflow: "hidden", background: "#000",
              boxShadow: SHADOW(), outline: on ? `4px solid ${hi()}` : "none", outlineOffset: 4 }}>
              {isClip(it.src) ? (
                <>
                  <Sequence from={at[i]} durationInFrames={clipF} layout="none"><OffthreadVideo src={staticFile(it.src)} muted style={media} /></Sequence>
                  {frame >= at[i] + clipF && it.last && <Img src={staticFile(it.last)} style={media} />}
                </>
              ) : <Img src={staticFile(it.src)} style={media} />}
              {it.stamp && <Tag text={it.stamp} />}
            </div>
          </div>
        );
      })}
    </>
  );
};

// ---------- quote ----------

// A sourced quote (<= 25 words), text only: no portrait, no logo. "mark" underlines one phrase of it on its cue.
export const Quote: React.FC<DP> = ({ v, words }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const a = calm(frame, 0, fps), b = calm(frame, 8, fps);
  const mp = ease(frame, cueAt(words, v.markCue, fps, Math.round(fps * 1.2)), 14);
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
        <span style={{ fontSize: 36, fontWeight: 800 }}>{v.name}</span>
        {v.role && <span style={{ fontSize: 30, fontWeight: 600, color: C.dim }}>{v.role}</span>}
      </div>
      {v.source && <div style={{ marginTop: 12, marginLeft: 64, fontSize: 25, color: C.dim, fontFamily: MONO, opacity: b }}>{v.source}</div>}
    </AbsoluteFill>
  );
};
