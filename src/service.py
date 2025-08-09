from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.video_editor import create_video_from_assets


# ----------------------------
# Configuration and constants
# ----------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT_DIR / "images"
AUDIO_DIR = ROOT_DIR / "audio"
WHISPER_DIR = ROOT_DIR / "whisper"
BACKGROUND_AUDIO_DIR = ROOT_DIR / "background_audio"
OUTPUT_DIR = ROOT_DIR

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "PoPHDFYHijTq7YiSCwE3")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "")

# Optional S3-compatible R2 config
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL", "")  # e.g. https://<accountid>.r2.cloudflarestorage.com
R2_BUCKET = os.getenv("R2_BUCKET", "faceless-yt")
R2_PUBLIC_DOMAIN = os.getenv("R2_PUBLIC_DOMAIN", "")  # e.g. https://pub-xxxxxx.r2.dev


# ----------------------------
# Request/Response models
# ----------------------------


class GenerateVideoRequest(BaseModel):
    name: str = Field(..., description="Person or topic name")
    biography: str = Field(..., description="Input biography text")
    system_prompt: Optional[str] = Field(
        default=(
            "You are a YouTube scriptwriter creating short, faceless documentary-style videos. "
            "Based on the biography provided, write a concise and factual script of around 275 words. "
            "The script will be used for a 1–2 minute YouTube Shorts video. Follow these rules: "
            "Tone: Neutral and documentary-style. Avoid dramatization or exaggerated storytelling. "
            "Content must be based only on verifiable facts from the input. Do not add fictionalized scenes, dialogue, or speculation. "
            "Do not make value judgments. Use short, clear sentences for voiceover. Follow a chronological structure (intro → milestones → factual closing line). "
            "After the closing line, add a short CTA like: If you enjoyed this, subscribe for more real stories from around the world."
        )
    )
    image_context: Optional[str] = Field(
        default=(
            "Photorealism Ultra Detailed 8K Ray Tracing Volumetric Lighting PBR Cinematic Composition. "
            "Design prompts suitable for faceless biography visuals."
        ),
        description="Stylistic guidance for image prompts",
    )
    group_interval_seconds: float = Field(6.0, ge=2.0, le=15.0, description="How many seconds of audio per image")
    output_basename: Optional[str] = Field(None, description="Override output video base filename (without extension)")
    upload_to_r2: bool = Field(False, description="Upload resulting video to R2 if configured")


class GenerateVideoResponse(BaseModel):
    title: Optional[str] = None
    caption: Optional[str] = None
    output_path: str
    output_url: Optional[str] = None
    images_generated: int
    audio_seconds: float


# ----------------------------
# Helpers
# ----------------------------


def _ensure_dirs() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    WHISPER_DIR.mkdir(parents=True, exist_ok=True)


async def _deepseek_chat(messages: List[Dict[str, Any]]) -> str:
    if not DEEPSEEK_API_KEY:
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY not configured")
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "deepseek-chat", "messages": messages, "stream": False}
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"Deepseek error: {r.text}")
        data = r.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


async def _generate_storyline(system_prompt: str, biography: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": biography},
    ]
    content = await _deepseek_chat(messages)
    # Minimal cleanup similar to the n8n flow step "Remove heading, title"
    cleanup_messages = [
        {
            "role": "system",
            "content": (
                "You will get a voice over text. Remove any heading or title. "
                "Return only the narration text with the final CTA kept. Target length ~250 words."
            ),
        },
        {"role": "user", "content": content},
    ]
    return await _deepseek_chat(cleanup_messages)


async def _elevenlabs_tts(text: str, voice_id: str = ELEVENLABS_VOICE_ID, filename: str = "voice.mp3") -> Path:
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=500, detail="ELEVENLABS_API_KEY not configured")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }
    body = {"text": text}
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(url, headers=headers, json=body)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"ElevenLabs error: {r.text}")
        audio_bytes = r.content
    out_path = AUDIO_DIR / filename
    out_path.write_bytes(audio_bytes)
    return out_path


async def _openai_whisper_transcribe(audio_path: Path) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    # multipart form
    files = {
        "file": (audio_path.name, audio_path.read_bytes(), "audio/mpeg"),
        "model": (None, "whisper-1"),
        "response_format": (None, "verbose_json"),
        "timestamp_granularities[]": (None, "word"),
    }
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(url, headers=headers, files=files)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"OpenAI Whisper error: {r.text}")
        return r.json()


def _group_words_by_interval(words: List[Dict[str, Any]], interval: float) -> List[Dict[str, Any]]:
    if not words:
        return []
    result: List[Dict[str, Any]] = []
    start_time = float(words[0]["start"])  # type: ignore[index]
    end_time = start_time + interval
    current_group: List[str] = []
    index = 0
    for w in words:
        w_start = float(w["start"])  # type: ignore[index]
        w_end = float(w["end"])  # type: ignore[index]
        token = str(w["word"])  # type: ignore[index]
        if w_start < end_time:
            current_group.append(token)
        else:
            index += 1
            result.append(
                {"text": " ".join(current_group).strip(), "start": start_time, "end": w_end, "index": index}
            )
            start_time = w_start
            end_time = start_time + interval
            current_group = [token]
    if current_group:
        index += 1
        last_end = float(words[-1]["end"])  # type: ignore[index]
        result.append({"text": " ".join(current_group).strip(), "start": start_time, "end": last_end, "index": index})
    return result


async def _deepseek_image_prompt(context_name: str, chunk_text: str, image_context: str) -> str:
    system = (
        "You are an image prompt designer. Convert the transcript text into a prompt for image generation. "
        f"Style and quality: {image_context}. Output only the prompt text. No quotes or commas."
    )
    user = f"Context - {context_name}. Image Prompt - {chunk_text}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = await _deepseek_chat(messages)
    # sanitize
    return (
        prompt.replace("\n", " ")
        .replace("\"", "")
        .replace(",", " ")
        .strip()
    )


async def _together_flux_image(prompt: str, width: int = 1024, height: int = 1024, steps: int = 4) -> bytes:
    if not TOGETHER_API_KEY:
        raise HTTPException(status_code=500, detail="TOGETHER_API_KEY not configured")
    url = "https://api.together.xyz/v1/images/generations"
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "black-forest-labs/FLUX.1-schnell",
        "prompt": prompt,
        "negative_prompt": (
            "text watermark signature paragraph wording letters symbols writing nude nudity explicit content "
            "obscene inappropriate offensive forbidden illegal prohibited sexual graphic violence gore blood disturbing"
        ),
        "width": width,
        "height": height,
        "steps": steps,
        "n": 1,
        "response_format": "b64_json",
    }
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"Together error: {r.text}")
        data = r.json()
    b64 = data["data"][0]["b64_json"]
    return base64.b64decode(b64)


def _write_image(idx: int, content: bytes, name_prefix: str) -> Path:
    filename = f"image_{idx:02d}_{name_prefix}.png"
    out_path = IMAGES_DIR / filename
    out_path.write_bytes(content)
    return out_path


def _seconds_from_whisper(result: Dict[str, Any]) -> float:
    # Prefer duration from last word if available
    if "words" in result:
        words = result["words"]
        if isinstance(words, list) and words:
            return float(words[-1].get("end", 0.0))
    # Fallback: segments
    if "segments" in result and isinstance(result["segments"], list) and result["segments"]:
        return float(result["segments"][-1].get("end", 0.0))
    return 0.0


# ----------------------------
# Optional R2 upload
# ----------------------------


def _maybe_upload_to_r2(local_path: Path, key: str) -> Optional[str]:
    if not (R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY and R2_ENDPOINT_URL and R2_BUCKET):
        return None
    try:
        import boto3  # lazy import

        s3 = boto3.client(
            "s3",
            endpoint_url=R2_ENDPOINT_URL,
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        )
        s3.upload_file(str(local_path), R2_BUCKET, key, ExtraArgs={"ContentType": "video/mp4", "ACL": "public-read"})
        if R2_PUBLIC_DOMAIN:
            return f"{R2_PUBLIC_DOMAIN.rstrip('/')}/{key}"
        # If no public domain, return s3 uri
        return f"s3://{R2_BUCKET}/{key}"
    except Exception as exc:  # noqa: BLE001
        # Do not fail pipeline on upload error
        print(f"R2 upload failed: {exc}")
        return None


# ----------------------------
# FastAPI app
# ----------------------------


app = FastAPI(title="Voxtale Video Service", version="1.0.0")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/videos", response_model=GenerateVideoResponse)
async def generate_video(payload: GenerateVideoRequest, background_tasks: BackgroundTasks) -> GenerateVideoResponse:  # noqa: D401
    """Generate a faceless biography video in one shot using our custom editor."""

    _ensure_dirs()

    # 1) Storyline
    storyline = await _generate_storyline(payload.system_prompt, payload.biography)

    # 2) ElevenLabs TTS
    audio_name = payload.output_basename or datetime.utcnow().strftime("%Y%m%d%H%M%S")
    audio_path = await _elevenlabs_tts(storyline, filename=f"{audio_name}.mp3")

    # 3) Whisper transcription (word timestamps)
    whisper_json = await _openai_whisper_transcribe(audio_path)
    whisper_path = WHISPER_DIR / "transcript.json"
    whisper_path.write_text(json.dumps(whisper_json, ensure_ascii=False), encoding="utf-8")

    words: List[Dict[str, Any]] = []
    # Normalize possible shapes
    if isinstance(whisper_json, dict) and "words" in whisper_json:
        words = whisper_json.get("words", [])
    elif isinstance(whisper_json, list) and whisper_json and "words" in whisper_json[0]:
        words = whisper_json[0].get("words", [])
    else:
        # If no words granularity, approximate from segments by splitting text
        segments = whisper_json.get("segments", []) if isinstance(whisper_json, dict) else []
        for seg in segments:
            seg_text = str(seg.get("text", "")).strip()
            start = float(seg.get("start", 0))
            end = float(seg.get("end", max(start + 2.0, start)))
            for token in seg_text.split():
                words.append({"word": token, "start": start, "end": end})

    groups = _group_words_by_interval(words, payload.group_interval_seconds)

    # 4) Generate images in parallel: Deepseek prompt -> Together FLUX
    async def gen_image(idx: int, text: str) -> Path:
        prompt = await _deepseek_image_prompt(payload.name, text, payload.image_context or "")
        img_bytes = await _together_flux_image(prompt)
        return _write_image(idx, img_bytes, audio_name)

    tasks = [gen_image(i + 1, g["text"]) for i, g in enumerate(groups) if g["text"]]
    image_paths = await asyncio.gather(*tasks)

    # 5) Create video
    output_path = OUTPUT_DIR / f"{audio_name}.mp4"

    def _render_video() -> None:
        create_video_from_assets(
            images_dir=str(IMAGES_DIR),
            audio_dir=str(AUDIO_DIR),
            transcript_path=str(whisper_path),
            output_path=str(output_path),
            fps=60,
            background_audio_dir=str(BACKGROUND_AUDIO_DIR),
            background_volume=0.3,
        )

    # MoviePy is synchronous/CPU-bound; run in threadpool to keep event loop responsive
    await asyncio.to_thread(_render_video)

    # 6) Optional upload to R2
    output_url: Optional[str] = None
    if payload.upload_to_r2:
        key = f"{audio_name}.mp4"
        output_url = _maybe_upload_to_r2(output_path, key)

    # Optionally: Title/Caption generation (kept minimal; can be extended)
    title = None
    caption = None

    return GenerateVideoResponse(
        title=title,
        caption=caption,
        output_path=str(output_path),
        output_url=output_url,
        images_generated=len(image_paths),
        audio_seconds=_seconds_from_whisper(whisper_json),
    )


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("src.service:app", host="0.0.0.0", port=8000, reload=False)


