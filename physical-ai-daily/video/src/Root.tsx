import React from "react";
import { Composition } from "remotion";
import { Episode, EpisodeProps } from "./Episode";

const FPS = 30;

// Props come from episodes/<dir>/props.json (written by tools/voice.py), passed with --props.
const placeholder: EpisodeProps = { title: "Physical AI Daily", date: "", segments: [] };

export const Root: React.FC = () => (
  <Composition
    id="Episode"
    component={Episode}
    width={1920}
    height={1080}
    fps={FPS}
    durationInFrames={FPS}
    defaultProps={placeholder}
    calculateMetadata={({ props }) => ({
      durationInFrames: Math.max(FPS, Math.ceil(props.segments.reduce((t, s) => t + s.duration, 0) * FPS)),
    })}
  />
);
