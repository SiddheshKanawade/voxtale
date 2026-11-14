export const C = {
  bg: "#0b0e14",
  panel: "#151a24",
  panel2: "#1d2432",
  ink: "#f2f4f8",
  dim: "#8b93a3",
  accent: "#ffe14d",
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

export type Beat = { narration: string; visual: Visual; audio: string; duration: number; words: Word[] };
export type Segment = {
  id: string; kind: "title" | "story"; headline: string; label?: string;
  source?: string; sourceDate?: string; beats: Beat[]; duration: number;
};
export type EpisodeProps = { title: string; date: string; segments: Segment[] };
