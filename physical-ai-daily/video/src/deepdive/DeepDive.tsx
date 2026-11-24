// Deep dive: one story, 12-20 min, told in chapters (cold open -> stakes -> before this -> the mechanism in 2-4 chapters
// -> results -> where it breaks -> what it opens -> recap). Entry src/deepdive/index.ts, composition "DeepDive";
// tools/deepdive/render.sh renders it, tools/deepdive/stills.sh renders review frames. props.json comes from
// tools/voice.py (unchanged): one take per chapter, cut into beats, with word timings.
//
// episode.json  {title, date, design?: "blueprint-light" (default) | "blueprint", voice, speed?, segments: Chapter[]}
// Chapter       {id, kind: "open" | "chapter" | "outro", headline, chapter?: n, question?, source?, sourceDate?,
//               beats: [{narration, visual, section?}]}. The sting plays after the first "open" segment; the end card
//               (visual "end") holds through the sonic logo. On "chapter" segments a small header ("03 · headline")
//               slides in once, on the first beat after the title card. Credit chip: visual.credit, else "Source: source".
//               section: a mono chip, top right (a key of ../Episode SECTIONS, or any short text).
// Cues          Every part waits for its cue: a word of THIS beat's narration (the first word starting with it, as
//               ../scenes cueFrame), "word#2" for its 2nd occurrence, or a number of seconds. Without one, parts spread
//               over the first 60% of the beat and the first shows within 0.4 s (a number's label holds the stage and
//               the number waits for its word). tools/deepdive/dd.py check lists cues that miss.
// cont: true    on build / loop / steps / timeline: continues the previous beat's diagram (no fade between the two;
//               parts cued at 0 are already drawn), so one diagram can grow over several beats.
// Coordinates   build: box centres in the stage, x 0-1680, y 0-660 (STAGE, between header and captions). figure: the
//               image's own width/height from its .json (the space of "highlight").
// Highlight     one at a time: the part just revealed, or a focus step, which also dims the rest. Red on paper,
//               yellow on black. Labels sit on the parts. Nothing moves except reveals, arrows drawing, a loop marker.
//
// Visual types (any other type falls through to ../scenes Scene: clip, shot, photo, kinetic, split, end, ...):
// chapter   title card, first beat of a chapter, <= 4 s: {number?, title?, question?, questionCue?}; defaults are the
//           segment's chapter, headline and question.
// build     the main teaching diagram, built part by part: {nodes: [{id, label, sub?, x, y, w?=400, h?, icon?, cue}],
//           edges?: [{from, to, label?, cue?, dashed?, loop?: true | "up", bend?: px, id?}], focus?: [{ids, cue}]}.
//           Boxes pop in (h defaults to fit their text); arrows draw from -> to, never before both boxes are in.
//           loop: an arc under (or over) both boxes, e.g. feedback; bend: curve off the straight line. focus ids: node
//           ids or edge ids ("from>to"); ids: [] clears. icon: agent code sim robot video build test check cross chart
//           data eye doc model person measure loop parts motor wall, or an emoji.
// figure    a paper figure, whole, then one region at a time: {src, regions?: [{x, y, w, h, cue, label?}], zoom?,
//           stamp?, credit}. Each cue dims the rest and frames the region, its label as a tab; zoom: straight cut-in.
// number    a headline number with its denominator: {value: "0 of 15", label, cue?, compare?: {value, label, cue},
//           note?, noteCue?, pips?: false}. "K of M" (M <= 60) also draws M squares, K filled. Digits on screen.
// predict   a prediction prompt: {question, options: [{label, sub?, cue?}] (2-3), answer: index, revealCue, tag?}.
//           Held while the viewer guesses; on revealCue the answer is marked and the rest dim.
// pair      two clips or images side by side: {items: [{src, label, stamp?, cue?}] x2, focus?: [{item, cue}]}. Clips
//           muted, last frame held. Credit: visual.credit, else the clips' own.
// quote     a sourced quote, <= 25 words, text only: {text, name, role?, source?, mark?, markCue?} (mark: a phrase of
//           text, underlined on its cue).
// timeline  a lineage, left to right: {items: [{year, label, sub?, cue?, now?}] x3-6}. The axis extends as items are
//           named; the current work ("now", else the last) takes the highlight and the rest step back.
// loop      an iterative cycle on a ring: {nodes: [{label, sub?, icon?, cue?}] x3-6, back?: {label?, cue?} (the closing
//           arrow), center?, centerCue?, cycleCue?, cycleS?}. On cycleCue a marker goes round once (0.55 s per node).
// steps     the same as a numbered row; back: {label?, cue?} adds a return arrow under it (and the marker's way back).
import React from "react";
import { AbsoluteFill, Audio, Sequence, staticFile, useCurrentFrame, useVideoConfig } from "remotion";
import { Beat, C, FONT, Segment, timeline } from "../theme";
import { END_S, STING_S, Sting } from "../brand";
import { applyDesign } from "../design";
import { Scene } from "../scenes";
import { Backdrop } from "../Episode";
import { Captions, ChapterHeader, Credit, Progress, SectionChip, TopBar, fadeInOut } from "./chrome";
import { Build, Cycle, Timeline } from "./diagrams";
import { ChapterCard, Figure, NumberCard, Pair, Predict, Quote } from "./cards";
import { DP } from "./kit";

export type DDSegment = Omit<Segment, "kind"> & { kind: "open" | "chapter" | "outro"; chapter?: number; question?: string };
export type DeepDiveProps = { title: string; date: string; segments: DDSegment[]; design?: string };

// The digest's timeline, with the sting after the cold open ("open" plays the role of the digest's "title").
export const ddTimeline = (segments: DDSegment[]) =>
  timeline(segments.map((s) => ({ ...s, kind: s.kind === "open" ? "title" : "story" }) as Segment), STING_S, END_S);

const DD: Record<string, React.FC<DP>> = {
  chapter: ChapterCard, build: Build, figure: Figure, number: NumberCard, predict: Predict, pair: Pair, quote: Quote, timeline: Timeline,
  loop: (p) => <Cycle {...p} ring />, steps: (p) => <Cycle {...p} ring={false} />,
};

// Our types render here; anything else (clip, shot, photo, kinetic, end...) falls through to the digest's Scene.
const DDScene: React.FC<DP & { joinNext: boolean }> = ({ joinNext, ...p }) => {
  const frame = useCurrentFrame();
  const Comp = DD[p.v.type];
  if (!Comp) return <Scene v={p.v} dur={p.dur} index={p.index} words={p.words} />;
  return <AbsoluteFill style={{ opacity: fadeInOut(frame, p.dur, !p.v.cont, !joinNext) }}><Comp {...p} /></AbsoluteFill>;
};

const NO_CREDIT = new Set(["chapter", "end"]);
const creditOf = (v: any, seg: DDSegment): string | null => {
  if (NO_CREDIT.has(v.type)) return null;
  const own = v.credit ?? (v.type === "pair" ? [...new Set((v.items ?? []).map((it: any) => it.credit).filter(Boolean))].join(" · ") : "");
  return own || (seg.source ? `Source: ${seg.source}${seg.sourceDate ? ` · ${seg.sourceDate}` : ""}` : null);
};

export const DeepDive: React.FC<DeepDiveProps> = ({ date, segments, design }) => {
  const look = applyDesign(design); // before any child renders: scenes read the colour tokens from C
  const { fps } = useVideoConfig();
  const tl = ddTimeline(segments);
  const endTail = Math.round(END_S * fps);
  const chapters = segments.filter((s) => s.kind === "chapter").length;
  let beatNo = 0;
  return (
    <AbsoluteFill style={{ fontFamily: FONT, color: C.ink, background: C.bg }}>
      <Backdrop kind={look.backdrop} />
      {segments.map((seg, k) => {
        let from = Math.round(tl.starts[k] * fps);
        const firstBody = seg.beats.findIndex((b) => b.visual.type !== "chapter"); // the header slides in here, once
        return seg.beats.map((beat: Beat, i) => {
          const start = from;
          const len = Math.round(beat.duration * fps);
          from += len;
          const n = beatNo++;
          const isLast = k === segments.length - 1 && i === seg.beats.length - 1;
          const shown = len + (isLast ? endTail : 0); // the end card holds through the sonic logo
          const v: any = beat.visual;
          const credit = creditOf(v, seg);
          return (
            <Sequence key={`${seg.id}-${i}`} from={start} durationInFrames={shown}>
              <Audio src={staticFile(beat.audio)} />
              <DDScene v={v} dur={shown} index={n} words={beat.words} seg={seg} chapters={chapters}
                joinNext={!!(seg.beats[i + 1]?.visual as any)?.cont} />
              {seg.kind === "chapter" && v.type !== "chapter" && <ChapterHeader n={seg.chapter} title={seg.headline} animate={i === firstBody} />}
              {beat.section && <SectionChip section={beat.section} />}
              {credit && <Credit text={credit} />}
              <Captions words={beat.words} />
            </Sequence>
          );
        });
      })}
      <TopBar date={date} />
      <Progress segments={segments} starts={tl.starts} total={tl.total} />
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
