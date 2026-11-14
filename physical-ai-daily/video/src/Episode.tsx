import React from "react";
import { AbsoluteFill, Audio, Sequence, spring, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { C, FONT, EpisodeProps, Segment, Word } from "./theme";
import { Scene } from "./scenes";

export type { EpisodeProps } from "./theme";

export const Episode: React.FC<EpisodeProps> = ({ title, date, segments }) => {
  const { fps } = useVideoConfig();
  const stories = segments.filter((s) => s.kind === "story");
  const total = segments.reduce((t, s) => t + s.duration, 0);
  let from = 0, beatNo = 0;
  return (
    <AbsoluteFill style={{ fontFamily: FONT, color: C.ink, background: C.bg }}>
      <Backdrop />
      {segments.map((seg) => {
        const segStart = from;
        return seg.beats.map((beat, i) => {
          const start = from;
          const len = Math.round(beat.duration * fps);
          from += len;
          const n = beatNo++;
          return (
            <Sequence key={`${seg.id}-${i}`} from={start} durationInFrames={len}>
              <Audio src={staticFile(beat.audio)} />
              <Scene v={beat.visual} dur={len} index={n} words={beat.words} />
              {seg.kind === "story" && <StoryHeader seg={seg} index={stories.indexOf(seg) + 1} total={stories.length} animate={start === segStart} />}
              {(beat.visual.credit || seg.source) && beat.visual.type !== "end" && (
                <Credit text={beat.visual.credit ?? `Source: ${seg.source} · ${seg.sourceDate}`} />
              )}
              <Captions words={beat.words} />
            </Sequence>
          );
        });
      })}
      <TopBar title={title} date={date} />
      <Chapters segments={segments} total={total} />
    </AbsoluteFill>
  );
};

// Slow-drifting dot grid so even static scenes never sit perfectly still.
const Backdrop: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill style={{
      background: `radial-gradient(ellipse at 25% 0%, #1b2436 0%, ${C.bg} 65%)`,
    }}>
      <AbsoluteFill style={{
        backgroundImage: "radial-gradient(rgba(255,255,255,0.07) 1.5px, transparent 1.5px)",
        backgroundSize: "44px 44px", backgroundPosition: `${frame * 0.3}px ${frame * 0.15}px`,
      }} />
    </AbsoluteFill>
  );
};

const TopBar: React.FC<{ title: string; date: string }> = ({ title, date }) => (
  <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 78, display: "flex", alignItems: "center",
    justifyContent: "space-between", padding: "0 64px", fontSize: 24, letterSpacing: 3, fontWeight: 700,
    background: "linear-gradient(rgba(11,14,20,0.9), rgba(11,14,20,0))" }}>
    <span><span style={{ color: C.accent }}>●</span> {title.toUpperCase()}</span>
    <span style={{ color: C.dim, fontWeight: 500, letterSpacing: 1 }}>{date}</span>
  </div>
);

// Chapter bar: one block per segment, filling as the episode plays, so viewers see how far along they are.
const Chapters: React.FC<{ segments: Segment[]; total: number }> = ({ segments, total }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  let acc = 0;
  return (
    <div style={{ position: "absolute", top: 70, left: 64, right: 64, height: 5, display: "flex", gap: 6 }}>
      {segments.map((s) => {
        const start = acc;
        acc += s.duration;
        const p = Math.min(1, Math.max(0, (t - start) / s.duration));
        return (
          <div key={s.id} style={{ flex: s.duration / total, background: "rgba(255,255,255,0.15)", borderRadius: 3, overflow: "hidden" }}>
            <div style={{ width: `${p * 100}%`, height: "100%", background: C.accent }} />
          </div>
        );
      })}
    </div>
  );
};

const StoryHeader: React.FC<{ seg: Segment; index: number; total: number; animate: boolean }> = ({ seg, index, total, animate }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = animate ? spring({ frame, fps, config: { damping: 200 } }) : 1;
  return (
    <>
      <div style={{ position: "absolute", top: 104, left: 64, display: "flex", alignItems: "center", gap: 18,
        opacity: s, transform: `translateX(${(1 - s) * -40}px)` }}>
        <span style={{ background: C.accent, color: "#111", fontWeight: 800, fontSize: 22, padding: "6px 12px",
          borderRadius: 6, letterSpacing: 2 }}>{seg.label}</span>
        <span style={{ fontSize: 42, fontWeight: 800, letterSpacing: -1, textShadow: "0 2px 12px rgba(0,0,0,0.6)" }}>{seg.headline}</span>
      </div>
      <div style={{ position: "absolute", top: 112, right: 64, fontSize: 26, color: C.dim, fontWeight: 700 }}>
        STORY {index}/{total}
      </div>
    </>
  );
};

const Credit: React.FC<{ text: string }> = ({ text }) => (
  <div style={{ position: "absolute", left: 64, bottom: 178, fontSize: 20, color: C.dim, fontWeight: 600,
    background: "rgba(11,14,20,0.8)", padding: "6px 14px", borderRadius: 8 }}>
    {text}
  </div>
);

// Groups words into short phrases, breaking at punctuation, and highlights the word being spoken.
const Captions: React.FC<{ words: Word[] }> = ({ words }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const now = frame / fps;
  const chunks: number[][] = [];
  let cur: number[] = [];
  words.forEach((w, i) => {
    cur.push(i);
    if (cur.length >= 7 || (/[.,?!:;]$/.test(w.text) && cur.length >= 2)) { chunks.push(cur); cur = []; }
  });
  if (cur.length) chunks.push(cur);
  let active = -1;
  for (let i = 0; i < words.length; i++) if (now >= words[i].start) active = i;
  if (active < 0 || now > words[words.length - 1].end + 0.3) return null;
  const chunk = chunks.find((c) => c.includes(active))!;
  return (
    <div style={{ position: "absolute", left: 0, right: 0, bottom: 64, display: "flex", justifyContent: "center" }}>
      <div style={{ background: "rgba(0,0,0,0.78)", borderRadius: 14, padding: "12px 26px", fontSize: 48, fontWeight: 700,
        letterSpacing: -0.5, maxWidth: 1640, textAlign: "center" }}>
        {chunk.map((i) => (
          <span key={i} style={{ color: i === active ? C.accent : C.ink, marginRight: 13 }}>{words[i].text}</span>
        ))}
      </div>
    </div>
  );
};
