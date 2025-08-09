from __future__ import annotations

import json
import random
import os
from pathlib import Path
from typing import List, Tuple

from moviepy import (
    AudioFileClip,
    ColorClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    concatenate_audioclips,
    concatenate_videoclips,
    vfx,
)

def _dynamic_image_durations(audio_duration: float, image_count: int) -> List[float]:
    """Return a list of durations for each image.
    
    The durations are calculated based on the audio duration and the number of images.
    """
    equally_distributed_duration = [audio_duration / image_count] * image_count
    for i in range(image_count):
        delta = round(random.uniform(0, (audio_duration / image_count) * 0.1), 2)
        j = random.randint(0, image_count - 1)
        if i == j:
            continue
        
        changed_i = equally_distributed_duration[i] - delta
        changed_j = equally_distributed_duration[j] + delta
        
        if changed_i <= 0.8*audio_duration / image_count or changed_j >= 1.2*audio_duration / image_count:
            continue
        
        equally_distributed_duration[i] = changed_i
        equally_distributed_duration[j] = changed_j
    
    return equally_distributed_duration

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
    if (isinstance(data, list) and len(data) > 0 and "words" in data[0]) or (
        isinstance(data, dict) and "words" in data
    ):
        # Word-level format: group words into segments
        words = data[0]["words"] if isinstance(data, list) else data["words"]

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
                len(current_segment) >= 10
                or word.endswith('.')
                or word.endswith('!')
                or word.endswith('?')
                or word.endswith(',')
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
                final_end = words[-1]["end"] if words else (segment_start + 1 if segment_start is not None else 1)
                captions.append(((segment_start if segment_start is not None else 0.0, final_end), text))
                
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


def _create_caption_clip(text: str, start: float, end: float, video_size):
    """Create a text clip with semi-transparent black background that fits the text.
    
    Args:
        text: The text to display
        start: Start time in seconds
        end: End time in seconds  
        video_size: Tuple of (width, height) for the video
    """
    duration = end - start
    video_width, video_height = video_size
    
    object_height_pixel = video_height - 150
    
    # Create the main text clip
    txt_clip = TextClip(
        text=text,
        font_size=38,
        color="white",
        stroke_color="black", 
        stroke_width=1,
        method="caption",
        size=(int(video_size[0] * 0.9), None),
        margin=(10, 10),
        text_align="center",
        transparent=True,
    ).with_start(start).with_duration(duration).with_position(("center", object_height_pixel))
    
    # Get the actual text dimensions
    text_width, text_height = txt_clip.size
    
    # Create black background that matches text size with padding
    bg_width = text_width+5  # 20px padding on each side
    bg_height = text_height+10  # 10px padding top/bottom
    
    background = ColorClip(
        size=(bg_width, bg_height),
        color=(0, 0, 0),  # Black
        duration=duration,
    ).with_opacity(0.5).with_start(start).with_position(("center", object_height_pixel)) # Position is top-left corner of the text
    
    return [background, txt_clip]


def _generate_caption_clip(captions: List[Tuple[Tuple[float, float], str]], video_size):
    """Create caption clips with semi-transparent backgrounds that fit the text.
    
    *captions* is a list of ((start, end), text) entries.
    Returns a list of clip objects for the composite video.
    """
    all_caption_clips = []
    
    for (start, end), text in captions:
        # Create caption clips for this text
        clips = _create_caption_clip(text, start, end, video_size)
        all_caption_clips.extend(clips)
    
    return all_caption_clips

def create_video_from_assets(
    images_dir: str | os.PathLike = "images",
    audio_dir: str | os.PathLike = "audio",
    transcript_path: str | os.PathLike = "whisper/transcript.json",
    output_path: str | os.PathLike = "output.mp4",
    image_effect: str = "kenburns",
    crossfade: float = 0.5,
    fps: int = 30,
    video_size: Tuple[int, int] = (1024, 1024),
    background_audio_dir: str | os.PathLike | None = "background_audio",
    background_volume: float = 0.3,
):
    """Create a narrated slideshow video with subtitles and optional background music.

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
        "kenburns" for alternating continuous zoom in/out effect, "none" for still images.
    crossfade
        Cross-fade duration in seconds between consecutive images.  Use 0 to
        disable.
    fps
        Frames per second for output video.
    background_audio_dir
        Folder containing background music file. If None, no background music is added.
        If the background audio is shorter than the video, it will be looped.
        If longer, it will be trimmed to match video duration.
    background_volume
        Volume level for background music (0.0 to 1.0). Default is 0.3.
    """

    images_dir = Path(images_dir)
    audio_dir = Path(audio_dir)
    transcript_path = Path(transcript_path)
    output_path = Path(output_path)

    images = _load_images(images_dir)
    audio_file = _select_audio(audio_dir)
    audio_clip = AudioFileClip(str(audio_file))

    # Determine how long each image stays on screen.
    img_durations = _dynamic_image_durations(audio_clip.duration, len(images))
    equally_distributed_duration = audio_clip.duration / len(images)
    # Normalize durations to exactly match the audio duration to avoid float drift
    current_total = sum(img_durations)
    if current_total <= 0:
        raise ValueError("Computed non-positive total duration for images")
    if abs(current_total - audio_clip.duration) > 1e-6:
        scale = audio_clip.duration / current_total
        img_durations = [d * scale for d in img_durations]
        # Correct any remaining rounding residue on the last item
        head_sum = sum(img_durations[:-1])
        img_durations[-1] = max(audio_clip.duration - head_sum, 0.0)

    video_clips: List[ImageClip] = []
    
    for i, img_path in enumerate(images):
        clip: ImageClip = ImageClip(str(img_path)).with_duration(img_durations[i])

        if image_effect == "kenburns":
            # Apply alternating continuous zoom in/out effect
            # Even indexed images zoom in, odd indexed images zoom out
            zoom_in = (i % 2 == 0)  # First image (index 0) zooms in, then alternates
            
            if zoom_in:
                # Zoom in: start at normal size, end at 1.3x
                clip = clip.with_effects([
                    vfx.Resize(lambda t: 1.0 + 0.3 * (t / equally_distributed_duration)) # Should have equally distributed duration of image, else 0.3/img_durations[i] can be less or greaterthan 0.3
                ])
            else:
                # Zoom out: start at 1.3x, end at normal size  
                clip = clip.with_effects([
                    vfx.Resize(lambda t: 1.3 - 0.3 * (t / equally_distributed_duration))
                ])
        clip = CompositeVideoClip([clip.with_position("center")], size=video_size)
        
        # Add cross-fade between consecutive clips using vfx
        if crossfade > 0 and i > 0:
            clip = clip.with_effects([vfx.CrossFadeIn(crossfade)])
        video_clips.append(clip)

    # Concatenate and add audio.
    slideshow = concatenate_videoclips(video_clips, method="compose")
    
    # Handle background music
    final_audio = audio_clip
    if background_audio_dir is not None:
        background_audio_dir = Path(background_audio_dir)
        try:
            background_audio_file = _select_audio(background_audio_dir)
            background_clip = AudioFileClip(str(background_audio_file))
            
            # Adjust background music volume
            background_clip = background_clip.with_volume_scaled(background_volume)
            
            # Handle duration differences
            video_duration = audio_clip.duration
            if background_clip.duration < video_duration:
                # Loop background music if it's shorter than video
                num_loops = int(video_duration / background_clip.duration) + 1
                background_clips = [background_clip] * num_loops
                looped_background = concatenate_audioclips(background_clips)
                background_clip = looped_background.subclipped(0, video_duration)
            elif background_clip.duration > video_duration:
                # Trim background music if it's longer than video
                background_clip = background_clip.subclipped(0, video_duration)
            
            # Mix narration with background music
            final_audio = CompositeAudioClip([audio_clip, background_clip])
            
        except FileNotFoundError as e:
            print(f"Warning: {e}. Proceeding without background music.")
    
    slideshow = slideshow.with_audio(final_audio)

    captions = _parse_whisper_transcript(transcript_path)
    subtitle_clips = _generate_caption_clip(captions, slideshow.size)

    # Combine slideshow with all subtitle clips
    all_clips = [slideshow] + subtitle_clips
    final_video = CompositeVideoClip(all_clips, size=video_size)

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
    parser.add_argument("--video_size", type=Tuple[int, int], default=(1024, 1024), help="Output video size")
    parser.add_argument("--background_audio_dir", default="background_audio", help="Folder with background music (optional)")
    parser.add_argument("--background_volume", type=float, default=0.3, help="Background music volume (0.0-1.0)")
    parser.add_argument("--no_background", action="store_true", help="Disable background music")
    args = parser.parse_args()

    create_video_from_assets(
        images_dir=args.images_dir,
        audio_dir=args.audio_dir,
        transcript_path=args.transcript,
        output_path=args.output,
        image_effect=args.effect,
        crossfade=args.crossfade,
        fps=args.fps,
        video_size=args.video_size,
        background_audio_dir=None if args.no_background else args.background_audio_dir,
        background_volume=args.background_volume,
    )


if __name__ == "__main__":
    _cli()
