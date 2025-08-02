from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Tuple

from moviepy import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    concatenate_videoclips,
    vfx,
)

def _load_images(image_dir: Path) -> List[Path]:
    """Return a sorted list of image paths from *image_dir*.

    Supports common image extensions (jpg, jpeg, png, webp).  Ignores non-files.
    """

    supported_ext = {".jpg", ".jpeg", ".png", ".webp"}
    images = [p for p in sorted(image_dir.iterdir()) if p.suffix.lower() in supported_ext]

    if not images:
        raise FileNotFoundError(f"No images found in {image_dir!s}")

    return images


def _select_audio(audio_dir: Path) -> Path:
    """Pick the first audio file found in *audio_dir* (wav, mp3, m4a, flac)."""
    supported_ext = {".wav", ".mp3", ".m4a", ".flac"}
    for p in sorted(audio_dir.iterdir()):
        if p.suffix.lower() in supported_ext and p.is_file():
            return p
    raise FileNotFoundError(f"No audio file found in {audio_dir!s}")


def _parse_whisper_transcript(json_path: Path) -> List[Tuple[Tuple[float, float], str]]:
    """Return a list of ((start, end), text) tuples from whisper *json_path*.

    This handles both word-level and segment-level Whisper transcript formats.
    """

    with open(json_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    captions = []
    
    # Handle different Whisper output formats
    if isinstance(data, list) and len(data) > 0 and "words" in data[0]:
        # Word-level format: group words into segments
        words = data[0]["words"]
        
        # Group words into segments (every 8-12 words or by punctuation)
        current_segment = []
        segment_start = None
        
        for word_data in words:
            word = word_data["word"]
            start = float(word_data["start"])
            end = float(word_data["end"])
            
            if segment_start is None:
                segment_start = start
            
            current_segment.append(word)
            
            # End segment on punctuation or after 10 words
            should_end_segment = (
                len(current_segment) >= 10 or
                word.endswith('.') or 
                word.endswith('!') or 
                word.endswith('?') or
                word.endswith(',')
            )
            
            if should_end_segment:
                text = " ".join(current_segment).strip()
                if text:  # Only add non-empty segments
                    captions.append(((segment_start, end), text))
                current_segment = []
                segment_start = None
        
        # Add any remaining words as final segment
        if current_segment:
            text = " ".join(current_segment).strip()
            if text:
                final_end = words[-1]["end"] if words else segment_start + 1
                captions.append(((segment_start, final_end), text))
                
    elif isinstance(data, dict) and "segments" in data:
        # Standard segment-level format
        for seg in data["segments"]:
            try:
                start = float(seg["start"])
                end = float(seg["end"])
                text = seg["text"].strip()
                if text:  # Only add non-empty segments
                    captions.append(((start, end), text))
            except (KeyError, ValueError) as e:
                raise ValueError(f"Bad segment entry: {seg}") from e
    else:
        raise ValueError("Unexpected Whisper JSON schema: expected 'segments' key or word-level format")

    return captions


def _generate_caption_clip(captions: List[Tuple[Tuple[float, float], str]], video_size):
    """Create caption clips from captions list.

    *captions* is a list of ((start, end), text) entries.
    Returns a list of TextClip objects positioned and timed correctly.
    """
    caption_clips = []
    
    for (start, end), text in captions:
        txt_clip = (
            TextClip(
                text=text,
                font_size=48,
                color="white",
                stroke_color="black",
                stroke_width=2,
                method="caption",
                size=(int(video_size[0] * 0.9), None),
            )
            .with_start(start)
            .with_duration(end - start)
            .with_position(("center", "bottom"))
        )
        caption_clips.append(txt_clip)
    
    return caption_clips



def create_video_from_assets(
    images_dir: str | os.PathLike = "images",
    audio_dir: str | os.PathLike = "audio",
    transcript_path: str | os.PathLike = "whisper/transcript.json",
    output_path: str | os.PathLike = "output.mp4",
    image_effect: str = "kenburns",
    crossfade: float = 0.5,
    fps: int = 30,
):
    """Create a narrated slideshow video with subtitles.

    Parameters
    ----------
    images_dir
        Folder containing ordered still images.  Images are taken in their
        *sorted* (alphabetical) order.
    audio_dir
        Folder containing a single audio track.  If multiple are present the
        first (alphabetically) is chosen.
    transcript_path
        Path to Whisper JSON transcript file.
    output_path
        Destination video file.  Container format is inferred from extension
        (e.g. `.mp4`, `.mov`, `.mkv`, ...).
    image_effect
        "kenburns" for slight zoom-in effect, "none" for still images.
    crossfade
        Cross-fade duration in seconds between consecutive images.  Use 0 to
        disable.
    fps
        Frames per second for output video.
    """

    images_dir = Path(images_dir)
    audio_dir = Path(audio_dir)
    transcript_path = Path(transcript_path)
    output_path = Path(output_path)

    images = _load_images(images_dir)
    audio_file = _select_audio(audio_dir)
    audio_clip = AudioFileClip(str(audio_file))

    # Determine how long each image stays on screen.
    img_duration = audio_clip.duration / len(images)

    video_clips: List[ImageClip] = []
    for i, img_path in enumerate(images):
        clip: ImageClip = ImageClip(str(img_path)).with_duration(img_duration)

        if image_effect == "kenburns":
            # Apply a Ken Burns effect using resize with a slight zoom
            # We'll resize the clip to be slightly larger and then crop it to create zoom effect
            zoom_factor = 1.1  # 10% zoom
            clip = clip.with_effects([
                vfx.Resize(zoom_factor),  # Zoom in by scaling up
            ])

        # Add cross-fade between consecutive clips using vfx
        if crossfade > 0 and i > 0:
            clip = clip.with_effects([vfx.CrossFadeIn(crossfade)])
        video_clips.append(clip)

    # Concatenate and add audio.
    slideshow = concatenate_videoclips(video_clips, method="compose")
    slideshow = slideshow.with_audio(audio_clip)

    captions = _parse_whisper_transcript(transcript_path)
    subtitle_clips = _generate_caption_clip(captions, slideshow.size)

    # Combine slideshow with all subtitle clips
    all_clips = [slideshow] + subtitle_clips
    final_video = CompositeVideoClip(all_clips)
    
    print(f"\n⏳ Rendering video to {output_path!s} (this may take a while)...")
    final_video.write_videofile(
        str(output_path),
        codec="libx264",
        audio_codec="aac",
        fps=fps,
        threads=os.cpu_count() or 4,
        temp_audiofile="temp-audio.m4a",
        remove_temp=True,
    )
    print("✅ Video creation complete.")


def _cli():  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Create slideshow video with captions.")
    parser.add_argument("--images_dir", default="images", help="Folder with input images")
    parser.add_argument("--audio_dir", default="audio", help="Folder with audio track")
    parser.add_argument(
        "--transcript", default="whisper/transcript.json", help="Path to Whisper JSON file"
    )
    parser.add_argument("--output", default="output.mp4", help="Output video path")
    parser.add_argument(
        "--effect",
        choices=["kenburns", "none"],
        default="kenburns",
        help="Image transition effect",
    )
    parser.add_argument(
        "--crossfade", type=float, default=0.5, help="Crossfade duration between images"
    )
    parser.add_argument("--fps", type=int, default=30, help="Output frames per second")

    args = parser.parse_args()

    create_video_from_assets(
        images_dir=args.images_dir,
        audio_dir=args.audio_dir,
        transcript_path=args.transcript,
        output_path=args.output,
        image_effect=args.effect,
        crossfade=args.crossfade,
        fps=args.fps,
    )


if __name__ == "__main__":
    _cli()
