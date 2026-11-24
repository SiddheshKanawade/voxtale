import React from "react";
import { Composition } from "remotion";
import { DeepDive, DeepDiveProps, ddTimeline } from "./DeepDive";

const FPS = 30;

// Props come from <dd-dir>/props.json (written by tools/voice.py), passed with --props. 1920x1080, rendered at 4K
// with --scale=2 (tools/deepdive/render.sh).
const placeholder: DeepDiveProps = { title: "Pixels2Actions deep dive", date: "", segments: [] };

export const DeepDiveRoot: React.FC = () => (
  <Composition
    id="DeepDive"
    component={DeepDive}
    width={1920}
    height={1080}
    fps={FPS}
    durationInFrames={FPS}
    defaultProps={placeholder}
    calculateMetadata={({ props }) => ({ durationInFrames: Math.max(FPS, Math.ceil(ddTimeline(props.segments).total * FPS)) })}
  />
);
