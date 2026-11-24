import React from "react";
import { spring, useCurrentFrame, useVideoConfig } from "remotion";
import { C, Word } from "../theme";
import { BRAND, Lockup } from "../brand";
import { current } from "../design";
import { SECTIONS } from "../Episode";
import { MONO, hi, pad2 } from "./kit";
import type { DDSegment } from "./DeepDive";

// The frame around a deep dive's scenes: top bar, chapter progress, chapter header, section chip, credit, captions.

// copied from ../Episode.tsx (kept independent)
export const TopBar: React.FC<{ date: string }> = ({ date }) => (
  <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 78, display: "flex", alignItems: "center",
    justifyContent: "space-between", padding: "0 64px", fontSize: 24, letterSpacing: 3, fontWeight: 700,
    background: `linear-gradient(rgba(${C.bgRgb},0.9), rgba(${C.bgRgb},0))` }}>
    <Lockup size={30} />
    <span style={{ color: C.dim, fontWeight: 500, letterSpacing: 1 }}>{date}</span>
  </div>
);

// copied from ../Episode.tsx Chapters (kept independent): one block per segment, filling as it plays. The track is
// ink on paper here (the digest's white track doesn't show on the light design).
export const Progress: React.FC<{ segments: DDSegment[]; starts: number[]; total: number }> = ({ segments, starts, total }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  return (
    <div style={{ position: "absolute", top: 70, left: 64, right: 64, height: 5, display: "flex", gap: 6 }}>
      {segments.map((s, k) => {
        const p = Math.min(1, Math.max(0, (t - starts[k]) / s.duration));
        return (
          <div key={s.id} style={{ flex: s.duration / total, borderRadius: 3, overflow: "hidden",
            background: current.light ? "rgba(31,32,35,0.12)" : "rgba(255,255,255,0.15)" }}>
            <div style={{ width: `${p * 100}%`, height: "100%", background: C.brand }} />
          </div>
        );
      })}
    </div>
  );
};

// Small, quiet "03 · How the agent fixes its robot", top left; slides in once per chapter, then holds.
export const ChapterHeader: React.FC<{ n?: number; title: string; animate: boolean }> = ({ n, title, animate }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = animate ? spring({ frame, fps, config: { damping: 200 } }) : 1;
  return (
    <div style={{ position: "absolute", top: 100, left: 64, display: "flex", alignItems: "baseline", gap: 14, fontSize: 28,
      fontWeight: 600, opacity: s, transform: `translateX(${(1 - s) * -30}px)` }}>
      {n != null && <span style={{ fontFamily: MONO, color: hi(), fontWeight: 600 }}>{pad2(n)}</span>}
      {n != null && <span style={{ color: C.dim }}>·</span>}
      <span style={{ color: C.ink, letterSpacing: -0.3, opacity: 0.85 }}>{title}</span>
    </div>
  );
};

// copied from ../Episode.tsx SectionChip (kept independent), mono style only (both blueprint designs use it).
export const SectionChip: React.FC<{ section: string }> = ({ section }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = spring({ frame, fps, config: { damping: 18 } });
  return (
    <div style={{ position: "absolute", top: 100, right: 64, padding: "8px 16px", fontSize: 22, fontWeight: 600, letterSpacing: 1,
      textTransform: "uppercase", borderRadius: 4, fontFamily: MONO, opacity: s, transform: `translateX(${(1 - s) * 60}px)`,
      ...(current.light
        ? { color: BRAND.red, background: "rgba(251,250,247,0.92)", border: "1px solid rgba(31,32,35,0.25)" }
        : { color: C.accent, background: "rgba(10,10,11,0.85)", border: "1px solid rgba(255,255,255,0.22)" }) }}>
      {`// ${SECTIONS[section] ?? section}`}
    </div>
  );
};

// copied from ../scenes.tsx Tag (kept independent): a quiet label on a source ("Simulation", "Preprint"). 26 px here
// (the digest's 22 px is hard to read on a phone).
export const Tag: React.FC<{ text: string; at?: { right: number; top: number } }> = ({ text, at = { right: 16, top: 14 } }) => (
  <div style={{ position: "absolute", ...at, fontSize: 26, fontWeight: 600, letterSpacing: 0.3,
    color: "#fff", background: "rgba(15,15,17,0.55)", padding: "5px 12px", borderRadius: 6 }}>{text}</div>
);

// copied from ../Episode.tsx (kept independent)
export const Credit: React.FC<{ text: string }> = ({ text }) => (
  <div style={{ position: "absolute", left: 64, bottom: 178, fontSize: 20, fontWeight: 600, padding: "6px 14px", borderRadius: 8,
    ...(current.light // a dark block reads as heavy on paper; keep the chip light there
      ? { color: "#55575e", background: "rgba(255,255,255,0.88)", border: "1px solid rgba(31,32,35,0.12)" }
      : { color: "#b5b3ae", background: "rgba(15,15,17,0.8)" }) }}>
    {text}
  </div>
);

// Adapted from ../Episode.tsx Captions (kept independent): the spoken word highlighted in a short phrase. Phrases never
// cross a sentence; inside one they break at clause punctuation or about 7 words, and a 1-2 word scrap is merged into
// its neighbour (a one-word page flashes past and reads as a glitch).
const SENT = /[.?!]["')\]]*$/, CLAUSE = /[,;:—–]["')\]]*$/;
export const phrases = (words: Word[]): number[][] => {
  const out: number[][] = [];
  let sent: number[] = [];
  const flush = () => {
    if (!sent.length) return;
    // clauses, then long clauses cut into near-equal runs of at most 7
    const parts: number[][] = [];
    let cl: number[] = [];
    sent.forEach((i, k) => { cl.push(i); if (CLAUSE.test(words[i].text) || k === sent.length - 1) { parts.push(cl); cl = []; } });
    const runs = parts.flatMap((c) => {
      const pieces = Math.ceil(c.length / 7), size = Math.ceil(c.length / pieces);
      return Array.from({ length: pieces }, (_, j) => c.slice(j * size, (j + 1) * size)).filter((r) => r.length);
    });
    // merge scraps (< 3 words) into a neighbour in the same sentence: the shorter side, so pages stay balanced
    for (let j = runs.findIndex((r) => r.length < 3); j >= 0 && runs.length > 1; j = runs.findIndex((r) => r.length < 3)) {
      const prev = j > 0 ? runs[j - 1].length : Infinity, next = j < runs.length - 1 ? runs[j + 1].length : Infinity;
      if (prev <= next) runs[j - 1].push(...runs[j]); else runs[j + 1].unshift(...runs[j]);
      runs.splice(j, 1);
    }
    out.push(...runs);
    sent = [];
  };
  words.forEach((w, i) => { sent.push(i); if (SENT.test(w.text)) flush(); });
  flush();
  return out;
};

export const Captions: React.FC<{ words: Word[] }> = ({ words }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const now = frame / fps;
  const chunks = React.useMemo(() => phrases(words), [words]);
  let active = -1;
  for (let i = 0; i < words.length; i++) if (now >= words[i].start) active = i;
  if (active < 0 || now > words[words.length - 1].end + 0.3) return null;
  const chunk = chunks.find((c) => c.includes(active))!;
  return (
    <div style={{ position: "absolute", left: 0, right: 0, bottom: 64, display: "flex", justifyContent: "center" }}>
      <div style={{ background: "rgba(0,0,0,0.78)", borderRadius: 14, padding: "12px 26px", fontSize: 48, fontWeight: 700,
        letterSpacing: -0.5, maxWidth: 1640, textAlign: "center" }}>
        {chunk.map((i) => (
          <span key={i} style={{ color: i === active ? BRAND.yellow : "#f5f4f2", marginRight: 13 }}>{words[i].text}</span>
        ))}
      </div>
    </div>
  );
};
