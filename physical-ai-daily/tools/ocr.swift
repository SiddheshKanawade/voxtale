import Vision
import AppKit
// Prints each recognized text line with its box as fractions of the image, top-left origin: x y w h | text
let url = URL(fileURLWithPath: CommandLine.arguments[1])
guard let img = NSImage(contentsOf: url), let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else { exit(1) }
let req = VNRecognizeTextRequest()
req.recognitionLevel = .accurate
try VNImageRequestHandler(cgImage: cg).perform([req])
for o in req.results ?? [] {
  guard let t = o.topCandidates(1).first else { continue }
  let b = o.boundingBox
  print(String(format: "%.4f %.4f %.4f %.4f | ", b.minX, 1 - b.maxY, b.width, b.height) + t.string)
}
