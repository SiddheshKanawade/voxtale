import { C } from "./theme";

// Channel design directions (docs/brand.md). An episode picks one with "design" in episode.json; the default is
// "charcoal". applyDesign() overwrites the shared colour tokens in C before the episode renders, so every scene follows.
export type Design = {
  colors: Partial<typeof C>;
  backdrop: "pixels" | "paper" | "blueprint" | "signal" | "aurora";
  chip: "outline" | "solid" | "mono" | "glass";
  light?: boolean; // dark text on a light background
  headlineShadow: string;
};

const DARK_TEXT = { ink: "#f5f4f2", dim: "#9a9893" };

export const DESIGNS: Record<string, Design> = {
  // Neutral charcoal, faint brand glows, drifting pixel grid (v6 proposal).
  charcoal: {
    colors: { bg: "#0f0f11", bgRgb: "15,15,17", panel: "#1a1a1d", panel2: "#242428", ...DARK_TEXT },
    backdrop: "pixels", chip: "outline", headlineShadow: "0 2px 12px rgba(0,0,0,0.6)",
  },
  // Light, like the brand's own site: warm paper, ink text, red and yellow accents.
  paper: {
    colors: { bg: "#fbfaf7", bgRgb: "251,250,247", panel: "#efede8", panel2: "#e6e3dc", ink: "#1f2023", dim: "#6b6d73", accent: "#d9a400" },
    backdrop: "paper", chip: "solid", headlineShadow: "none",
  },
  // Lab notebook: near-black with a fine, static engineering grid and monospaced section labels. The user's pick
  // (2026-11-14). The grid doesn't move: constant background motion competes with the content (docs/attention.md).
  blueprint: {
    colors: { bg: "#0a0a0b", bgRgb: "10,10,11", panel: "#151517", panel2: "#1e1e21", ...DARK_TEXT },
    backdrop: "blueprint", chip: "mono", headlineShadow: "0 2px 12px rgba(0,0,0,0.6)",
  },
  // The same lab notebook on warm paper: ink grid, ink text, red section labels. Dark text on a light background reads
  // better (positive-polarity advantage) and matches the mostly white source screenshots.
  "blueprint-light": {
    colors: { bg: "#fbfaf7", bgRgb: "251,250,247", panel: "#efede8", panel2: "#e6e3dc", ink: "#1f2023", dim: "#6b6d73", accent: "#c99a00" },
    backdrop: "blueprint", chip: "mono", light: true, headlineShadow: "none",
  },
  // Bold: black with a brand-red edge and a red light sweep; solid red section labels.
  signal: {
    colors: { bg: "#0c0c0d", bgRgb: "12,12,13", panel: "#18181a", panel2: "#222225", ...DARK_TEXT },
    backdrop: "signal", chip: "solid", headlineShadow: "0 2px 12px rgba(0,0,0,0.6)",
  },
  // Modern: slow, blurred red and yellow light drifting behind frosted-glass labels.
  aurora: {
    colors: { bg: "#0d0d10", bgRgb: "13,13,16", panel: "rgba(255,255,255,0.06)", panel2: "rgba(255,255,255,0.10)", ...DARK_TEXT },
    backdrop: "aurora", chip: "glass", headlineShadow: "0 2px 16px rgba(0,0,0,0.5)",
  },
};

const BASE = { ...C };

export let current: Design = DESIGNS.charcoal;

export const applyDesign = (name?: string): Design => {
  current = DESIGNS[name ?? "charcoal"] ?? DESIGNS.charcoal;
  Object.assign(C, BASE, current.colors);
  return current;
};
