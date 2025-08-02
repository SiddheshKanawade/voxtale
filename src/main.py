from __future__ import annotations

from pathlib import Path

from src.video_editor import create_video_from_assets


def main() -> None:
    root = Path(__file__).resolve().parent.parent

    create_video_from_assets(
        images_dir=root / "images",
        audio_dir=root / "audio",
        transcript_path=root / "whisper" / "transcript.json",
        output_path=root / "output.mp4",
    )


if __name__ == "__main__":
    main()

