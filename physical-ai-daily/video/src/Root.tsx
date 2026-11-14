import React from "react";
import { Composition } from "remotion";
import { Episode, EpisodeProps } from "./Episode";
import { END_S, STING_S } from "./brand";
import { timeline } from "./theme";

const FPS = 30;

// Props come from episodes/<dir>/props.json (written by tools/voice.py), passed with --props.
const placeholder: EpisodeProps = { title: "Pixels2Actions", date: "", segments: [] };

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
      durationInFrames: Math.max(FPS, Math.ceil(timeline(props.segments, STING_S, END_S).total * FPS)),
    })}
  />
);
