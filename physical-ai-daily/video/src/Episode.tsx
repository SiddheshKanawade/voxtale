import React from "react";
import { AbsoluteFill, Audio, Sequence, interpolate, spring, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { C, FONT, EpisodeProps, Segment, Word, timeline } from "./theme";
import { BRAND, END_S, Lockup, STING_S, Sting } from "./brand";
import { Design, applyDesign, current } from "./design";
import { Scene } from "./scenes";

export type { EpisodeProps } from "./theme";

export const Episode: React.FC<EpisodeProps> = ({ date, segments, design, reviewLabel }) => {
  const look = applyDesign(design); // before any child renders: scenes read the colour tokens from C
  const { fps } = useVideoConfig();
  const stories = segments.filter((s) => s.kind === "story");
  const tl = timeline(segments, STING_S, END_S);
  const endTail = Math.round(END_S * fps);
  let beatNo = 0;
  return (
    <AbsoluteFill style={{ fontFamily: FONT, color: C.ink, background: C.bg }}>
      <Backdrop kind={look.backdrop} />
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
              {/* skip the chip when the scene's own title already names the section */}
              {beat.section && SECTIONS[beat.section]?.toLowerCase() !== beat.visual.title?.toLowerCase() && <SectionChip section={beat.section} />}
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
      {reviewLabel && ( // design-comparison clips only: names the option on screen
        <div style={{ position: "absolute", right: 60, bottom: 170, background: "rgba(0,0,0,0.8)", color: "#fff", fontSize: 56,
          fontWeight: 800, padding: "14px 28px", borderRadius: 14 }}>{reviewLabel}</div>
      )}
      <Sequence from={Math.round(tl.end * fps)} durationInFrames={endTail}>
        <Audio src={staticFile("brand/sonic_logo.wav")} volume={0.5} />
      </Sequence>
    </AbsoluteFill>
  );
};

const PIXEL_TILE_OF = (fill: string, opacity: number) => `url("data:image/svg+xml,${encodeURIComponent(
  `<svg xmlns="http://www.w3.org/2000/svg" width="44" height="44"><rect x="19" y="19" width="6" height="6" rx="1.2" fill="${fill}" fill-opacity="${opacity}"/></svg>`)}")`;
const PIXEL_TILE = PIXEL_TILE_OF("#fff", 0.05);

const PAPER_TILE = PIXEL_TILE_OF("#1f2023", 0.06);
const GRID = (step: number, a: number, rgb = "255,255,255") =>
  `linear-gradient(rgba(${rgb},${a}) 1px, transparent 1px) 0 0 / ${step}px ${step}px, ` +
  `linear-gradient(90deg, rgba(${rgb},${a}) 1px, transparent 1px) 0 0 / ${step}px ${step}px`;

// Backdrops. The pixel, signal and aurora candidates drift; blueprint, the chosen one, stays still (docs/attention.md).
export const Backdrop: React.FC<{ kind: Design["backdrop"] }> = ({ kind }) => {
  const frame = useCurrentFrame();
  const drift = `${frame * 0.3}px ${frame * 0.15}px`;
  if (kind === "paper") return (
    <AbsoluteFill style={{ background: `radial-gradient(ellipse 60% 55% at 8% 0%, rgba(244,63,46,0.06), transparent 70%), ${C.bg}` }}>
      <AbsoluteFill style={{ backgroundImage: PAPER_TILE, backgroundSize: "44px 44px", backgroundPosition: drift }} />
    </AbsoluteFill>
  );
  if (kind === "blueprint") return current.light ? (  // static grid: no drift (docs/attention.md)
    <AbsoluteFill style={{ background: C.bg }}>
      <AbsoluteFill style={{ background: `${GRID(240, 0.09, "31,32,35")}, ${GRID(48, 0.045, "31,32,35")}` }} />
    </AbsoluteFill>
  ) : (
    <AbsoluteFill style={{ background: `radial-gradient(ellipse 120% 90% at 50% 45%, transparent 55%, rgba(0,0,0,0.5)), ${C.bg}` }}>
      <AbsoluteFill style={{ background: `${GRID(240, 0.09)}, ${GRID(48, 0.05)}` }} />
    </AbsoluteFill>
  );
  if (kind === "signal") {
    const x = 70 + 25 * Math.sin(frame / 90);
    return (
      <AbsoluteFill style={{ background: [
        `radial-gradient(ellipse 45% 70% at ${x}% 110%, rgba(244,63,46,0.22), transparent 70%)`,
        "radial-gradient(ellipse 120% 90% at 50% 45%, transparent 55%, rgba(0,0,0,0.5))", C.bg].join(", ") }}>
        <AbsoluteFill style={{ backgroundImage: PIXEL_TILE, backgroundSize: "44px 44px", backgroundPosition: drift }} />
        <div style={{ position: "absolute", left: 0, top: 0, bottom: 0, width: 10, background: BRAND.red }} />
      </AbsoluteFill>
    );
  }
  if (kind === "aurora") {
    const blob = (color: string, x: number, y: number, size: number) => (
      <div style={{ position: "absolute", left: x - size / 2, top: y - size / 2, width: size, height: size, borderRadius: "50%",
        background: color, filter: "blur(150px)", opacity: 0.32 }} />
    );
    return (
      <AbsoluteFill style={{ background: C.bg, overflow: "hidden" }}>
        {blob(BRAND.red, 380 + 220 * Math.sin(frame / 140), 160 + 120 * Math.cos(frame / 170), 900)}
        {blob("rgba(240,201,69,0.55)", 1550 + 200 * Math.cos(frame / 160), 900 + 100 * Math.sin(frame / 130), 700)}
        <AbsoluteFill style={{ background: "radial-gradient(ellipse 120% 90% at 50% 45%, transparent 50%, rgba(0,0,0,0.55))" }} />
      </AbsoluteFill>
    );
  }
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
    background: `linear-gradient(rgba(${C.bgRgb},0.9), rgba(${C.bgRgb},0))` }}>
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
        <span style={{ fontSize: 42, fontWeight: 800, letterSpacing: -1, textShadow: current.headlineShadow }}>{seg.headline}</span>
      </div>
      <div style={{ position: "absolute", top: 112, right: 64, fontSize: 26, color: C.dim, fontWeight: 700 }}>
        STORY {index}/{total}
      </div>
    </>
  );
};

// Named sections recur in every story, so viewers learn the structure: each gets the same chip. Visual only: a tick
// sound would overlap the narration, and sounds over speech hurt recall (docs/attention.md).
export const SECTIONS: Record<string, string> = {
  plain: "In plain English", evidence: "The evidence", matters: "Why it matters", reality: "Reality check",
};

const CHIP: Record<Design["chip"], (glow: number) => React.CSSProperties> = {
  outline: (g) => ({ background: "rgba(26,26,29,0.92)", border: `2px solid rgba(244,63,46,${0.35 + 0.65 * g})`, color: "#f5f4f2" }),
  solid: (g) => ({ background: BRAND.red, color: "#fff", boxShadow: `0 0 ${30 * g}px rgba(244,63,46,0.8)` }),
  mono: () => ({ fontFamily: "'SF Mono', Menlo, monospace", fontWeight: 600, letterSpacing: 1, borderRadius: 4,
    ...(current.light
      ? { color: BRAND.red, background: "rgba(251,250,247,0.92)", border: "1px solid rgba(31,32,35,0.25)" }
      : { color: C.accent, background: "rgba(10,10,11,0.85)", border: "1px solid rgba(255,255,255,0.22)" }) }),
  glass: (g) => ({ background: "rgba(255,255,255,0.08)", backdropFilter: "blur(14px)", color: "#f5f4f2",
    border: `1px solid rgba(255,255,255,${0.18 + 0.4 * g})` }),
};

const SectionChip: React.FC<{ section: string }> = ({ section }) => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  const s = spring({ frame, fps, config: { damping: 18 } });
  const glow = interpolate(frame, [0, 18], [1, 0], { extrapolateRight: "clamp" });
  return (
    <div style={{ position: "absolute", top: 156, right: 64, display: "flex", alignItems: "center", gap: 12,
        padding: "8px 16px", fontSize: 22, fontWeight: 800, letterSpacing: 3, textTransform: "uppercase", borderRadius: 10,
        opacity: s, transform: `translateX(${(1 - s) * 60}px)`, ...CHIP[current.chip](glow) }}>
        {current.chip !== "mono" && <span style={{ width: 14, height: 14, borderRadius: 3, background: BRAND.yellow, display: "inline-block" }} />}
        {current.chip === "mono" ? `// ${SECTIONS[section] ?? section}` : SECTIONS[section] ?? section}
    </div>
  );
};

const Credit: React.FC<{ text: string }> = ({ text }) => (
  <div style={{ position: "absolute", left: 64, bottom: 178, fontSize: 20, fontWeight: 600, padding: "6px 14px", borderRadius: 8,
    ...(current.light // a dark block reads as heavy on paper; keep the chip light there
      ? { color: "#55575e", background: "rgba(255,255,255,0.88)", border: "1px solid rgba(31,32,35,0.12)" }
      : { color: "#b5b3ae", background: "rgba(15,15,17,0.8)" }) }}>
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
          <span key={i} style={{ color: i === active ? BRAND.yellow : "#f5f4f2", marginRight: 13 }}>{words[i].text}</span>
        ))}
      </div>
    </div>
  );
};
