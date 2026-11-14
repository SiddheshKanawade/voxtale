import React from "react";
import { Composition, Still } from "remotion";
import { Episode, EpisodeProps } from "./Episode";
import { END_S, STING_S } from "./brand";
import { timeline } from "./theme";
import { Thumbnail, ThumbProps } from "./Thumbnail";

const FPS = 30;

// Props come from episodes/<dir>/props.json (written by tools/voice.py), passed with --props.
const placeholder: EpisodeProps = { title: "Pixels2Actions", date: "", segments: [] };

const thumbPlaceholder: ThumbProps = { layout: "face", lines: ["Robot brain", "goes to *work*"] };

export const Root: React.FC = () => (
  <>
  <Still id="Thumbnail" component={Thumbnail} width={1280} height={720} defaultProps={thumbPlaceholder} />
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
</>
);
