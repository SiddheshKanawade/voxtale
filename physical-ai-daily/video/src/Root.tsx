import React from "react";
import { Composition, Still } from "remotion";
import { Episode, EpisodeProps } from "./Episode";
import { ChannelIcon, END_S, Lockup, LogoMark, STING_S } from "./brand";
import { timeline } from "./theme";
import { Thumbnail, ThumbProps } from "./Thumbnail";
import { Banner, BANNER, BannerProps } from "./Banner";
import { applyDesign } from "./design";

const FPS = 30;

// Props come from episodes/<dir>/props.json (written by tools/voice.py), passed with --props.
const placeholder: EpisodeProps = { title: "Pixels2Actions", date: "", segments: [] };

const thumbPlaceholder: ThumbProps = { layout: "face", lines: ["Robot brain", "goes to *work*"] };

export const Root: React.FC = () => (
  <>
  <Still id="Thumbnail" component={Thumbnail} width={1280} height={720} defaultProps={thumbPlaceholder} />
  {/* Brand exports (tools/brand_assets.sh): profile picture, the mark on transparent, the lockup on light. */}
  <Still id="ChannelIcon" component={ChannelIcon} width={800} height={800} />
  <Still id="LogoMark" component={() => <LogoMark size={1024} />} width={1024} height={1024} />
  <Still id="LockupLight" component={() => { applyDesign("blueprint-light"); return (
    <div style={{ width: "100%", height: "100%", background: "#f7f5f0", display: "flex", alignItems: "center", justifyContent: "center" }}>
      <Lockup size={150} />
    </div>
  ); }} width={1600} height={400} />
  <Still id="ChannelBanner" component={Banner} width={BANNER.width} height={BANNER.height} defaultProps={{ thumbs: [] } as BannerProps} />
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
