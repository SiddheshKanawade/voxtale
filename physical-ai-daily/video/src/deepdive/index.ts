import { registerRoot } from "remotion";
import { DeepDiveRoot } from "./DeepDiveRoot";

// Separate entry for the long-form deep dives, so the digest's composition (src/index.ts) is untouched:
// npx remotion render src/deepdive/index.ts DeepDive out.mp4 --props=<dd>/props.json --public-dir=<dd>
registerRoot(DeepDiveRoot);
