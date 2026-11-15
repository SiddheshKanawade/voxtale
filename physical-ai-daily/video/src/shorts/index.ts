import { registerRoot } from "remotion";
import { ShortsRoot } from "./ShortsRoot";

// Separate entry for 9:16 Shorts, so the episode composition (src/index.ts) is untouched:
// npx remotion render src/shorts/index.ts Short out.mp4 --props=<short>/props.json --public-dir=<short>
registerRoot(ShortsRoot);
