// Deep dive: one story, 12-20 min, told in chapters (cold open -> stakes -> before this -> the mechanism in 2-4 chapters
// -> results -> where it breaks -> what it opens -> recap). Entry src/deepdive/index.ts, composition "DeepDive";
// tools/deepdive/render.sh renders it, tools/deepdive/stills.sh renders review frames. props.json comes from
// tools/voice.py (unchanged): one take per chapter, cut into beats, with word timings.
//
// episode.json  {title, date, design?: "blueprint-light" (default) | "blueprint", voice, speed?, segments: Chapter[]}
// Chapter       {id, kind: "open" | "chapter" | "outro", headline, chapter?: n, question?, source?, sourceDate?,
//               beats: [{narration, visual, section?, pause?}]}. The sting plays after the first "open" segment; the end card
//               (visual "end") holds through the sonic logo. On "chapter" segments a small header ("03 · headline")
//               slides in once, on the first beat after the title card. Credit chip: visual.credit, else "Source: source".
//               section: a mono chip, top right (a key of ../Episode SECTIONS, or any short text).
// pause         seconds of silence after a beat's narration (voice.py keeps the field): the visual holds, nothing is said,
//               then the next beat. For a prediction: options on screen, revealCue "pause" ticks the answer in the
//               pause's last second (after the viewer had time to guess); "pause+1.5" is 1.5 s into it.
// Cuts          Never through blank paper, never onto an empty slot: the next visual starts 3 frames before its beat and
//               is in by +6 while the outgoing one holds full until its beat ends, then fades over 6. Every visual
//               arrives settled, its first content element already in place whatever its cue (the first node, bar,
//               tile, timeline item, a number card's number), and later ones build onto it. The video's first frame
//               shows the first visual whole. The sting dissolves in over the cold open's last visual and out onto the
//               next one. Consecutive grids of the same shape, and consecutive split pairs, swap clips in place (a slot
//               keeps its old clip until the new one's word; a clip that stays keeps playing); "join": false cuts.
// Cues          Every later part waits for its cue: a word of THIS beat's narration (the first word starting with it, as
//               ../scenes cueFrame), "word#2" for its 2nd occurrence, a number of seconds, "pause" or "pause+X". Without
//               one, parts spread over the first 60% of the narration. Hold rule: every cue lands by the beat's end
//               (pause included) - 1 s, so each finished build is seen whole for 1 s. tools/deepdive/dd.py check lists
//               cues that miss, cues within 1 s of the end, and first elements shown well before their word.
// cont: true    on build / loop / steps / timeline / bars: continues the previous beat's visual of the same type, with no
//               cut: parts cued at 0 are already in place, the highlight carries over, a build chain keeps one fit.
// Layout        build: box centres x, y in stage units (about 1680 x 660); build / loop / steps are auto-fitted to ~80%
//               of the stage (text 38 px titles, 28 px subs at scale 1, never past 64 px). Nothing marks a part not yet
//               named ("scaffold": true draws dashed outlines, for a deliberate fill-in-the-blanks). Tags, chips and the
//               credit are >= 28 px. figure regions: the image's own width/height from its .json (the space of "highlight").
// Highlight     one at a time: the part just revealed, or a focus step, which also dims the rest. Red on paper,
//               yellow on black. Labels sit on the parts. Nothing moves except reveals, arrows drawing, a loop marker.
//
// Visual types (any other type falls through to ../scenes Scene: clip, shot, photo, kinetic, split, end, ...):
// chapter   title card, first beat of a chapter, <= 4 s: {number?, title?, question?, questionCue?}; defaults are the
//           segment's chapter, headline and question.
// build     the main teaching diagram, built part by part: {nodes: [{id, label, sub?, x, y, w?=400, h?, icon?, cue}],
//           edges?: [{from, to, label?, cue?, dashed?, loop?: true | "up", bend?: px, id?}], focus?: [{ids, cue}],
//           scaffold?}. Boxes pop in (h fits the text); arrows draw from -> to, never before both boxes are in.
//           loop: an arc under (or over) both boxes, e.g. feedback; bend: curve off the straight line. focus ids: node
//           ids or edge ids ("from>to"); ids: [] clears. icon: agent code sim robot video build test check cross chart
//           data eye doc model person measure loop parts motor wall, or an emoji.
// figure    a paper figure, whole, then one region at a time: {src, regions?: [{x, y, w, h, cue, label?}], zoom?,
//           stamp?, credit}. Each cue dims the rest and frames the region, its label as a tab; zoom: straight cut-in.
// number    a headline number with its denominator: {value: "0 of 15", label, cue?, compare?: {value, label, cue},
//           note?, noteCue?, pips?: false}. The number and its label are in place at the cut (a label never shows alone);
//           the comparison lands on its word. "K of M" (M <= 60) also draws M squares, K filled. Digits on screen.
// bars      a bar chart: {bars: [{label, value, display?, cue, note?, range?: [lo, hi], denominator?}], max?,
//           baseline?: {value, label?, cue?}, orient?: "v" (default) | "h", title?}. The first bar is in place at the
//           cut; each next one grows on its word with its category, its number (display, e.g. "73%" or "20–60%") at
//           the bar's end and denominator ("57 of 86"). range: a dashed low-high span. max: 100 for percentages.
// predict   a prediction prompt: {question, options: [{label, sub?, cue?}] (2-3), answer: index, revealCue, tag?}.
//           Give the beat a "pause" (2-3 s) and revealCue "pause": the question is asked, the options hold through the
//           silence while the viewer guesses, then the answer is marked and the rest dim.
// video     one clip as a shot (the digest's clip can't crop or align): {src, crop?: {x, y, w, h} (clip pixels,
//           punched in to fill the frame), from? / to? (clip seconds), playbackRate?, freeze?, check?, travel?, label?,
//           outcome?, ok?, stamp?, credit?}. Past "to" or the clip's end, the last frame holds.
// pair      sim vs real: {items: [side, side], focus?: [{item, cue}], freeze?: {cue, s=1.5, at?: [secA, secB], marks?:
//           [{side, x, y, shape?: "ring" | "arrow", label?}]}, mode?: "split" | "sequence", split?}. A side: {src, label,
//           outcome?, ok?, outcomeCue?, stamp?, crop?, from?, playbackRate?, to?, check?: {cue, ok?, x?, y?}, travel?:
//           "right" | {dir, x, y}, reveal?}. Split (default): equal frames, both at full opacity throughout (reveal: hold
//           a side back to its word); focus is a border and label colour on the side the narration is on, from the
//           start of the clause that names it, never a dimming. from / playbackRate align the two clips so they diverge
//           at once; on freeze.cue both hold (at freeze.at, else where they are) for s seconds and red marks land on the
//           divergence (x, y in that side's clip pixels). check: a ✓ (or ✗, ok: false) lands on its cue, e.g. on the
//           last frame past the last bump. outcome: "Simulation ✓ crosses" / "Real ✗ drops off". mode "sequence": side A
//           full frame, then side B full frame on items[1].cue, then the split on "split" (each a cut; each clip starts
//           when it comes on). Clips muted. Credit: visual.credit, else the clips' own.
// grid      2x2 or 2x3 clips or images: {items: [{src, label, stamp?, cue?, crop?, from?, playbackRate?}] x4-6, focus?:
//           [{item, cue}], cols?, fit?: "contain" | "cover"}. The first tile is in place at the cut, the others land on
//           their cues; labelled on the tile; one focus at a time (border and label colour). Real outcomes as footage.
// quote     a sourced quote, <= 25 words, text only: {text, name, role?, source?, mark?, markCue?} (mark: a phrase of
//           text, underlined on its cue).
// timeline  a lineage, left to right: {items: [{year, label, sub?, cue?, now?}] x3-6}. The axis extends as items are
//           named; the current work ("now", else the last) takes the highlight and the rest step back.
// loop      an iterative cycle on a ring: {nodes: [{label, sub?, icon?, cue?}] x3-6, back?: {label?, cue?} (the closing
//           arrow), center?, centerCue?, cycleCue?, cycleS?}. On cycleCue a marker goes round once (0.55 s per node,
//           finishing by the hold).
// steps     the same as a numbered row; back: {label?, cue?} adds a return arrow under it (and the marker's way back).
import React from "react";
import { AbsoluteFill, Audio, Sequence, interpolate, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { Beat, C, FONT, Segment, timeline } from "../theme";
import { END_S, STING_S, Sting } from "../brand";
import { applyDesign } from "../design";
import { Scene } from "../scenes";
import { Backdrop } from "../Episode";
import { Captions, ChapterHeader, Credit, Progress, SectionChip, TopBar } from "./chrome";
import { Bars, Build, Cycle, Timeline } from "./diagrams";
import { ChapterCard, Figure, NumberCard, Predict, Quote } from "./cards";
import { Grid, Pair, Video } from "./media";
import { DP, SETTLED, clamp } from "./kit";

export type DDSegment = Omit<Segment, "kind"> & { kind: "open" | "chapter" | "outro"; chapter?: number; question?: string };
export type DeepDiveProps = { title: string; date: string; segments: DDSegment[]; design?: string };

// A beat's "pause" (seconds of silence after its narration, the visual holding) counts in its duration on screen.
const withPauses = (segments: DDSegment[]): DDSegment[] => segments.map((s) => {
  const beats = s.beats.map((b) => ({ ...b, duration: b.duration + ((b as any).pause ?? 0) }));
  return { ...s, beats, duration: beats.reduce((a, b) => a + b.duration, 0) };
});

// The digest's timeline, with the sting after the cold open ("open" plays the role of the digest's "title"), on
// durations that include the pauses.
export const ddTimeline = (segments: DDSegment[]) =>
  timeline(withPauses(segments).map((s) => ({ ...s, kind: s.kind === "open" ? "title" : "story" }) as Segment), STING_S, END_S);

const DD: Record<string, React.FC<DP>> = {
  chapter: ChapterCard, build: Build, figure: Figure, number: NumberCard, predict: Predict, pair: Pair, grid: Grid, video: Video, quote: Quote,
  timeline: Timeline, bars: Bars, loop: (p) => <Cycle {...p} ring />, steps: (p) => <Cycle {...p} ring={false} />,
};

// Cut timing, in frames. A visual is on screen from `pre` frames before its beat to `tail` frames after it.
const PRE = 3, FADE_IN = 9, TAIL = 6; // crossfade: in from -3 to +6, the outgoing one full until its end, out by +6
const FIRST_PRE = 15; // a digest Scene first in the video: its own slide-in is over by frame 0
const STING_SKIP = 4, STING_IN = 4, AFTER_STING = 8; // the sting starts on its tile, not on its flat first frames

type Slot = { p: DP; start: number; len: number; last: boolean; tail: number; fadeIn: boolean; fadeOut: boolean; beat: Beat; seg: DDSegment; i: number };

// Every beat's place on the cut timeline: lead-in, tail, fades, the cont chain it belongs to.
const plan = (segments: DDSegment[], fps: number, starts: number[], endTail: number, stingK: number): Slot[] => {
  const chapters = segments.filter((s) => s.kind === "chapter").length;
  const flat = segments.flatMap((seg, k) => {
    let from = Math.round(starts[k] * fps);
    return seg.beats.map((beat, i) => {
      const start = from, len = Math.round(beat.duration * fps), pause = Math.round(((beat as any).pause ?? 0) * fps);
      from += len;
      return { seg, k, i, beat, start, len, pause, v: beat.visual as any };
    });
  });
  // A beat joins the one before (no crossfade, nothing rebuilt) when it says "cont", or when both are grids of the same
  // shape or both split pairs: those swap their clips in place. Never across the sting.
  const sameLayout = (a: any, b: any) =>
    (a.type === "grid" && b.type === "grid" && (a.items?.length ?? 0) === (b.items?.length ?? 0) && (a.cols ?? 0) === (b.cols ?? 0))
    || (a.type === "pair" && b.type === "pair" && a.mode !== "sequence" && b.mode !== "sequence");
  const afterSting = (n: number) => n > 0 && flat[n].k === stingK + 1 && flat[n].i === 0;
  const joins = (n: number) => n > 0 && n < flat.length && !afterSting(n) && !!DD[flat[n].v.type] && flat[n].v.type === flat[n - 1].v.type
    && flat[n].v.join !== false && (!!flat[n].v.cont || sameLayout(flat[n - 1].v, flat[n].v));
  const slots: Slot[] = flat.map((b, n) => {
    const dd = !!DD[b.v.type], last = n === flat.length - 1, cont = joins(n), joinNext = joins(n + 1);
    const beforeSting = b.k === stingK && b.i === b.seg.beats.length - 1;
    const pre = n === 0 ? (dd ? 0 : FIRST_PRE) : cont ? 0 : afterSting(n) ? AFTER_STING : PRE;
    const tail = last ? endTail : joinNext ? 0 : beforeSting ? STING_IN : TAIL;
    const words = b.beat.words.map((w) => ({ ...w, start: w.start + pre / fps, end: w.end + pre / fps }));
    // every visual arrives settled: the cut's crossfade brings its first state in whole (no entrance from blank)
    const p: DP = { v: b.v, dur: pre + b.len + tail, index: n, words, seg: b.seg, chapters, pre, len: b.len - b.pause, pause: b.pause,
      hold: Math.max(pre, pre + b.len - fps), born: SETTLED };
    return { p, start: b.start, len: b.len, last, tail, beat: b.beat, seg: b.seg, i: b.i,
      fadeIn: n > 0 && !cont && !afterSting(n), fadeOut: !last && !joinNext && !beforeSting };
  });
  slots.forEach((s, n) => {
    if (!joins(n) && !joins(n + 1)) return;
    if (joins(n)) s.p.prev = slots[n - 1].p;
    let a = n, z = n;
    while (joins(a)) a--;
    while (joins(z + 1)) z++;
    s.p.chain = flat.slice(a, z + 1).map((b) => b.v);
  });
  return slots;
};

// Our types render here; anything else (clip, shot, photo, kinetic, end...) falls through to the digest's Scene, whose
// own 6-frame fade-in runs inside the lead-in (and whose fade-out is pushed past the end where there should be none).
const DDScene: React.FC<{ s: Slot }> = ({ s }) => {
  const frame = useCurrentFrame();
  const { p } = s;
  const Comp = DD[p.v.type];
  if (!Comp) return <Scene v={p.v} dur={p.dur + (s.fadeOut ? 0 : 6)} index={p.index} words={p.words} />;
  const o = (s.fadeIn ? interpolate(frame, [0, FADE_IN], [0, 1], clamp) : 1) * (s.fadeOut ? interpolate(frame, [p.dur - s.tail, p.dur], [1, 0], clamp) : 1);
  return <AbsoluteFill style={{ opacity: o }}><Comp {...p} /></AbsoluteFill>;
};

// The digest's sting, joined in on its tile (the flat first frames skipped) with a short dissolve over the last shot.
const StingCut: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();
  return (
    <AbsoluteFill style={{ opacity: interpolate(frame, [0, STING_IN], [0, 1], clamp) }}>
      <Sequence from={-STING_SKIP} durationInFrames={Math.round(STING_S * fps) + STING_SKIP}><Sting /></Sequence>
    </AbsoluteFill>
  );
};

const NO_CREDIT = new Set(["chapter", "end"]);
const creditOf = (v: any, seg: DDSegment): string | null => {
  if (NO_CREDIT.has(v.type)) return null;
  const own = v.credit ?? (v.type === "video" ? v.src?.credit
    : v.type === "pair" || v.type === "grid" ? [...new Set((v.items ?? []).map((it: any) => it.credit).filter(Boolean))].join(" · ") : "");
  return own || (seg.source ? `Source: ${seg.source}${seg.sourceDate ? ` · ${seg.sourceDate}` : ""}` : null);
};

export const DeepDive: React.FC<DeepDiveProps> = ({ date, segments, design }) => {
  const look = applyDesign(design); // before any child renders: scenes read the colour tokens from C
  const { fps } = useVideoConfig();
  const tl = ddTimeline(segments);
  const endTail = Math.round(END_S * fps);
  const stingK = tl.sting >= 0 ? segments.findIndex((s) => s.kind === "open") : -1;
  const segs = withPauses(segments);
  const slots = plan(segs, fps, tl.starts, endTail, stingK);
  const stingF = Math.round(tl.sting * fps), stingLen = Math.round(STING_S * fps);
  return (
    <AbsoluteFill style={{ fontFamily: FONT, color: C.ink, background: C.bg }}>
      <Backdrop kind={look.backdrop} />
      {/* visuals first, each from its lead-in to its tail, so a later one draws over the one it replaces */}
      {slots.map((s) => (
        <Sequence key={`v${s.p.index}`} from={s.start - s.p.pre} durationInFrames={s.p.dur}>
          <DDScene s={s} />
        </Sequence>
      ))}
      {/* then each beat's sound and chrome, on the beat itself, above every visual */}
      {slots.map((s) => {
        const v = s.p.v, credit = creditOf(v, s.seg);
        const firstBody = s.seg.beats.findIndex((b) => b.visual.type !== "chapter"); // the header slides in here, once
        return (
          <Sequence key={`b${s.p.index}`} from={s.start} durationInFrames={s.len + (s.last ? endTail : 0)}>
            <Audio src={staticFile(s.beat.audio)} />
            {s.seg.kind === "chapter" && v.type !== "chapter" && <ChapterHeader n={s.seg.chapter} title={s.seg.headline} animate={s.i === firstBody} />}
            {s.beat.section && <SectionChip section={s.beat.section} />}
            {credit && <Credit text={credit} />}
            <Captions words={s.beat.words} />
          </Sequence>
        );
      })}
      <TopBar date={date} />
      <Progress segments={segs} starts={tl.starts} total={tl.total} />
      {tl.sting >= 0 && (
        <>
          <Sequence from={stingF} durationInFrames={stingLen}><StingCut /></Sequence>
          {/* the sonic logo keeps its sync with the pixel pops: it starts where the skipped frames would have */}
          <Sequence from={stingF - STING_SKIP} durationInFrames={stingLen + STING_SKIP}>
            <Audio src={staticFile("brand/sonic_logo.wav")} volume={0.5} />
          </Sequence>
        </>
      )}
      <Sequence from={Math.round(tl.end * fps)} durationInFrames={endTail}>
        <Audio src={staticFile("brand/sonic_logo.wav")} volume={0.5} />
      </Sequence>
    </AbsoluteFill>
  );
};
