import React from "react";
import { AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { C, FONT, EpisodeProps, Segment, Word, timeline } from "./theme";
import { BRAND, END_S, Lockup, STING_S, Sting } from "./brand";
import { Scene } from "./scenes";

export type { EpisodeProps } from "./theme";

export const Episode: React.FC<EpisodeProps> = ({ date, segments }) => {
  const { fps } = useVideoConfig();
  const stories = segments.filter((s) => s.kind === "story");
  const tl = timeline(segments, STING_S, END_S);
  const endTail = Math.round(END_S * fps);
  let beatNo = 0;
  return (
    <AbsoluteFill style={{ fontFamily: FONT, color: C.ink, background: C.bg }}>
      <Backdrop />
      {segments.map((seg, k) => {
        let from = Math.round(tl.starts[k] * fps);
        const segStart = from;
        return seg.beats.map((beat, i) => {
          const start = from;
          const len = Math.round(beat.duration * fps);
          from += len;
          const n = beatNo++;
          const isLast = k === segments.length - 1 && i === seg.beats.length - 1;
          const shown = len + (isLast ? endTail : 0); // the end card holds through the sonic logo
          return (
            <Sequence key={`${seg.id}-${i}`} from={start} durationInFrames={shown}>
              <Audio src={staticFile(beat.audio)} />
              <Scene v={beat.visual} dur={shown} index={n} words={beat.words} />
              {beat.section && <SectionChip section={beat.section} hideChip={SECTIONS[beat.section]?.toLowerCase() === beat.visual.title?.toLowerCase()} />}
              {seg.kind === "story" && <StoryHeader seg={seg} index={stories.indexOf(seg) + 1} total={stories.length} animate={start === segStart} />}
              {(beat.visual.credit || seg.source) && beat.visual.type !== "end" && (
                <Credit text={beat.visual.credit ?? `Source: ${seg.source} · ${seg.sourceDate}`} />
              )}
              <Captions words={beat.words} />
            </Sequence>
          );
        });
      })}
      <TopBar date={date} />
      <Chapters segments={segments} starts={tl.starts} total={tl.total} />
      {tl.sting >= 0 && (
        <Sequence from={Math.round(tl.sting * fps)} durationInFrames={Math.round(STING_S * fps)}>
          <Sting />
          <Audio src={staticFile("brand/sonic_logo.wav")} volume={0.5} />
        </Sequence>
      )}
      <Sequence from={Math.round(tl.end * fps)} durationInFrames={endTail}>
        <Audio src={staticFile("brand/sonic_logo.wav")} volume={0.5} />
      </Sequence>
    </AbsoluteFill>
  );
};

const PIXEL_TILE = `url("data:image/svg+xml,${encodeURIComponent(
  '<svg xmlns="http://www.w3.org/2000/svg" width="44" height="44"><rect x="19" y="19" width="6" height="6" rx="1.2" fill="#fff" fill-opacity="0.05"/></svg>')}")`;

// Neutral charcoal with a faint warm glow in the brand colours, a vignette, and a slow-drifting grid of square
// "pixels" (the brand motif) so even static scenes never sit perfectly still.
const Backdrop: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill style={{
      background: [
        "radial-gradient(ellipse 60% 55% at 8% 0%, rgba(244,63,46,0.11), transparent 70%)",
        "radial-gradient(ellipse 55% 50% at 100% 100%, rgba(240,201,69,0.06), transparent 70%)",
        "radial-gradient(ellipse 120% 90% at 50% 45%, transparent 55%, rgba(0,0,0,0.45))",
        C.bg,
      ].join(", "),
    }}>
      <AbsoluteFill style={{
        backgroundImage: PIXEL_TILE,
        backgroundSize: "44px 44px", backgroundPosition: `${frame * 0.3}px ${frame * 0.15}px`,
      }} />
    </AbsoluteFill>
  );
};

const TopBar: React.FC<{ date: string }> = ({ date }) => (
  <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 78, display: "flex", alignItems: "center",
    justifyContent: "space-between", padding: "0 64px", fontSize: 24, letterSpacing: 3, fontWeight: 700,
    background: "linear-gradient(rgba(15,15,17,0.9), rgba(15,15,17,0))" }}>
    <Lockup size={30} />
    <span style={{ color: C.dim, fontWeight: 500, letterSpacing: 1 }}>{date}</span>
  </div>
);

// Chapter bar: one block per segment, filling as the episode plays, so viewers see how far along they are.
const Chapters: React.FC<{ segments: Segment[]; starts: number[]; total: number }> = ({ segments, starts, total }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const t = frame / fps;
  return (
    <div style={{ position: "absolute", top: 70, left: 64, right: 64, height: 5, display: "flex", gap: 6 }}>
      {segments.map((s, k) => {
        const start = starts[k];
        const p = Math.min(1, Math.max(0, (t - start) / s.duration));
        return (
          <div key={s.id} style={{ flex: s.duration / total, background: "rgba(255,255,255,0.15)", borderRadius: 3, overflow: "hidden" }}>
            <div style={{ width: `${p * 100}%`, height: "100%", background: C.brand }} />
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
        <span style={{ background: C.brand, color: "#fff", fontWeight: 800, fontSize: 22, padding: "6px 12px",
          borderRadius: 6, letterSpacing: 2 }}>{seg.label}</span>
        <span style={{ fontSize: 42, fontWeight: 800, letterSpacing: -1, textShadow: "0 2px 12px rgba(0,0,0,0.6)" }}>{seg.headline}</span>
      </div>
      <div style={{ position: "absolute", top: 112, right: 64, fontSize: 26, color: C.dim, fontWeight: 700 }}>
        STORY {index}/{total}
      </div>
    </>
  );
};

// Named sections recur in every story, so viewers learn the structure: each gets the same chip and tick sound.
export const SECTIONS: Record<string, string> = {
  plain: "In plain English", evidence: "The evidence", matters: "Why it matters", reality: "Reality check",
};

// hideChip: the scene already shows the section name as its title, so only the tick plays.
const SectionChip: React.FC<{ section: string; hideChip?: boolean }> = ({ section, hideChip }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = spring({ frame, fps, config: { damping: 18 } });
  const glow = interpolate(frame, [0, 18], [1, 0], { extrapolateRight: "clamp" });
  return (
    <>
      <Audio src={staticFile("brand/tick.wav")} volume={0.35} />
      {!hideChip && <div style={{ position: "absolute", top: 156, right: 64, display: "flex", alignItems: "center", gap: 12,
        background: "rgba(26,26,29,0.92)", border: `2px solid rgba(244,63,46,${0.35 + 0.65 * glow})`, borderRadius: 10,
        padding: "8px 16px", fontSize: 22, fontWeight: 800, letterSpacing: 3, textTransform: "uppercase",
        opacity: s, transform: `translateX(${(1 - s) * 60}px)` }}>
        <span style={{ width: 14, height: 14, borderRadius: 3, background: BRAND.yellow, display: "inline-block" }} />
        {SECTIONS[section] ?? section}
      </div>}
    </>
  );
};

const Credit: React.FC<{ text: string }> = ({ text }) => (
  <div style={{ position: "absolute", left: 64, bottom: 178, fontSize: 20, color: C.dim, fontWeight: 600,
    background: "rgba(15,15,17,0.8)", padding: "6px 14px", borderRadius: 8 }}>
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
