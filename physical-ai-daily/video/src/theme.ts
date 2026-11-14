export const C = {
  bg: "#0f0f11", // neutral charcoal (was navy; user found it bluish and dated)
  bgRgb: "15,15,17", // bg as r,g,b for fades
  panel: "#1a1a1d",
  panel2: "#242428",
  ink: "#f5f4f2",
  dim: "#9a9893",
  accent: "#f0c945", // Pixels2Actions yellow
  brand: "#f43f2e", // Pixels2Actions red
  good: "#4ade80",
  bad: "#f87171",
  warn: "#fbbf24",
};

export const FONT = "-apple-system, 'SF Pro Display', 'Helvetica Neue', Arial, sans-serif";

// Stage: the area between the story header and the captions where scenes draw.
export const STAGE = { x: 120, y: 190, w: 1680, h: 660 };

export type Media = { src: string; width: number; height: number; highlight: null | { x: number; y: number; w: number; h: number } };
export type Word = { text: string; start: number; end: number };

export type Visual = {
  type: string;
  src?: Media;
  side?: Media;
  fit?: "cover" | "contain";
  pan?: "in" | "out" | "left" | "right";
  overlay?: string;
  badge?: string;
  stamp?: string;
  tags?: string[];
  credit?: string;
  text?: string;
  sub?: string;
  title?: string;
  items?: any[];
  steps?: { icon: string; label: string; sub?: string }[];
  letters?: { letter: string; word: string }[];
  center?: { icon: string; label: string };
  spokes?: { icon: string; label: string; sub?: string }[];
  left?: { icon: string; title: string; sub: string };
  right?: { icon: string; title: string; sub: string };
};

export type Beat = { narration: string; visual: Visual; audio: string; duration: number; words: Word[]; section?: string };
export type Segment = {
  id: string; kind: "title" | "story"; headline: string; label?: string;
  source?: string; sourceDate?: string; beats: Beat[]; duration: number;
};
export type EpisodeProps = { title: string; date: string; segments: Segment[]; design?: string; reviewLabel?: string };

// Segment start times in seconds, with the brand sting (stingS long) inserted after the cold-open title segment and
// an end tail (endS) after the last segment, where the end card holds while the sonic logo plays.
export const timeline = (segments: Segment[], stingS: number, endS = 0) => {
  let t = 0, sting = -1;
  const starts = segments.map((s, i) => {
    const at = t;
    t += s.duration;
    if (s.kind === "title" && sting < 0 && i < segments.length - 1) { sting = t; t += stingS; }
    return at;
  });
  return { starts, sting, end: t, total: t + endS };
};
