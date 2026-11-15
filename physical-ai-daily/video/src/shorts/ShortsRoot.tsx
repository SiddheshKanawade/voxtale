import React from "react";
import { Composition } from "remotion";
import { END_S, Short, ShortProps } from "./Short";

const FPS = 30;
const placeholder: ShortProps = { date: "", hook: "Hook", segments: [] };

export const ShortsRoot: React.FC = () => (
  <Composition
    id="Short"
    component={Short}
    width={1080}
    height={1920}
    fps={FPS}
    durationInFrames={FPS}
    defaultProps={placeholder}
    calculateMetadata={({ props }) => ({
      durationInFrames: Math.max(FPS, Math.ceil((props.segments.reduce((a, s) => a + s.duration, 0) + END_S) * FPS)),
    })}
  />
);
