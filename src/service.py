from __future__ import annotations

import asyncio
import base64
import json
import os
import logging
import time
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.video_editor import create_video_from_assets

load_dotenv()


# ----------------------------
# Logging
# ----------------------------

logger = logging.getLogger("voxtale.service")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s - %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


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
    logger.info("Ensuring directories exist under root: %s", ROOT_DIR)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    WHISPER_DIR.mkdir(parents=True, exist_ok=True)


async def _deepseek_chat(messages: List[Dict[str, Any]]) -> str:
    if not DEEPSEEK_API_KEY:
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY not configured")
    url = "https://api.deepseek.com/chat/completions"
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": "deepseek-chat", "messages": messages, "stream": False}
    logger.info("Calling Deepseek chat with %d messages", len(messages))
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            logger.error("Deepseek error: %s", r.text)
            raise HTTPException(status_code=502, detail=f"Deepseek error: {r.text}")
        data = r.json()
    result = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
    logger.info("Deepseek chat response length=%d", len(result))
    return result


async def _generate_storyline(system_prompt: str, biography: str) -> str:
    logger.info("Generating storyline from biography (length=%d)", len(biography))
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": biography},
    ]
    content = await _deepseek_chat(messages)
    logger.info("Initial storyline length=%d", len(content))
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
    final_storyline = await _deepseek_chat(cleanup_messages)
    logger.info("Final storyline length=%d", len(final_storyline))
    return final_storyline


async def _elevenlabs_tts(text: str, voice_id: str = ELEVENLABS_VOICE_ID, filename: str = "voice.mp3") -> Path:
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=500, detail="ELEVENLABS_API_KEY not configured")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json",
    }
    body = {"text": text}
    logger.info("Requesting ElevenLabs TTS (text_len=%d, voice=%s, filename=%s)", len(text), voice_id, filename)
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(url, headers=headers, json=body)
        if r.status_code >= 400:
            logger.error("ElevenLabs error: %s", r.text)
            raise HTTPException(status_code=502, detail=f"ElevenLabs error: {r.text}")
        audio_bytes = r.content
    out_path = AUDIO_DIR / filename
    out_path.write_bytes(audio_bytes)
    logger.info("Saved TTS audio (%d bytes) to %s", len(audio_bytes), out_path)
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
    logger.info("Transcribing audio with Whisper: %s", audio_path)
    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(url, headers=headers, files=files)
        if r.status_code >= 400:
            logger.error("OpenAI Whisper error: %s", r.text)
            raise HTTPException(status_code=502, detail=f"OpenAI Whisper error: {r.text}")
        result = r.json()
        logger.info("Whisper transcription received")
        return result


def _group_words_by_interval(words: List[Dict[str, Any]], interval: float) -> List[Dict[str, Any]]:
    if not words:
        return []
    logger.info("Grouping %d words by interval %.2f", len(words), interval)
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
    logger.info("Formed %d text groups", len(result))
    return result


async def _deepseek_image_prompt(context_name: str, chunk_text: str, image_context: str) -> str:
    system = (
        "You are an image prompt designer. Convert the transcript text into a prompt for image generation. "
        f"Style and quality: {image_context}. Output only the prompt text. No quotes or commas."
    )
    user = f"Context - {context_name}. Image Prompt - {chunk_text}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    logger.info("Generating image prompt (context=%s, chunk_len=%d)", context_name, len(chunk_text))
    prompt = await _deepseek_chat(messages)
    # sanitize
    sanitized = (
        prompt.replace("\n", " ")
        .replace("\"", "")
        .replace(",", " ")
        .strip()
    )
    logger.info("Image prompt generated (len=%d)", len(sanitized))
    return sanitized


async def _together_flux_image(prompt: str, width: int = 1440, height: int = 960, steps: int = 4) -> bytes:
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
    logger.info("Requesting FLUX image (w=%d h=%d steps=%d)", width, height, steps)

    # Simple 1 QPS rate limiter with retries for Together API
    together_qps_env = os.getenv("TOGETHER_QPS", "0.25")
    try:
        together_qps = max(float(together_qps_env), 0.1)
    except Exception:
        together_qps = 1.0
    min_interval = 1.0 / together_qps

    # Shared state for throttling
    if not hasattr(_together_flux_image, "_last_ts"):
        _together_flux_image._last_ts = 0.0  # type: ignore[attr-defined]
        _together_flux_image._lock = asyncio.Lock()  # type: ignore[attr-defined]

    max_attempts = 6
    for attempt in range(max_attempts):
        # Respect QPS
        async with _together_flux_image._lock:  # type: ignore[attr-defined]
            now = time.monotonic()
            elapsed = now - _together_flux_image._last_ts  # type: ignore[attr-defined]
            if elapsed < min_interval:
                wait_s = min_interval - elapsed
                logger.debug("Together QPS throttle: sleeping %.2fs", wait_s)
                await asyncio.sleep(wait_s)
            _together_flux_image._last_ts = time.monotonic()  # type: ignore[attr-defined]

        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(url, headers=headers, json=payload)

        # Handle errors with retry/backoff on rate limiting
        if r.status_code >= 400:
            text = r.text
            is_rate_limited = (r.status_code == 429) or ("rate_limit" in text.lower())
            if is_rate_limited and attempt < max_attempts - 1:
                backoff = min(2 ** attempt, 30) + random.uniform(0, 0.5)
                logger.warning(
                    "Together rate limited (attempt %d/%d). Backing off for %.2fs", attempt + 1, max_attempts, backoff
                )
                await asyncio.sleep(backoff)
                continue

            logger.error("Together FLUX error (status=%d): %s", r.status_code, text)
            raise HTTPException(status_code=502, detail=f"Together error: {text}")

        data = r.json()
        try:
            b64 = data["data"][0]["b64_json"]
        except Exception:
            logger.error("Unexpected Together response schema: %s", data)
            raise HTTPException(status_code=502, detail=f"Together error: {data}")

        content = base64.b64decode(b64)
        logger.info("FLUX image received (%d bytes)", len(content))
        return content

    # Should not reach here due to raise in loop
    raise HTTPException(status_code=502, detail="Together error: rate limit retries exhausted")


def _write_image(idx: int, content: bytes, name_prefix: str) -> Path:
    filename = f"image_{idx:02d}_{name_prefix}.png"
    out_path = IMAGES_DIR / filename
    out_path.write_bytes(content)
    logger.info("Wrote image #%d to %s (%d bytes)", idx, out_path, len(content))
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
            region_name="auto",
        )
        logger.info("Uploading to R2 bucket=%s key=%s", R2_BUCKET, key)
        s3.upload_file(str(local_path), R2_BUCKET, key, ExtraArgs={"ContentType": "video/mp4", "ACL": "public-read"})
        if R2_PUBLIC_DOMAIN:
            url = f"{R2_PUBLIC_DOMAIN.rstrip('/')}/{key}"
            logger.info("R2 upload complete: %s", url)
            return url
        # If no public domain, return s3 uri
        url = f"s3://{R2_BUCKET}/{key}"
        logger.info("R2 upload complete: %s", url)
        return url
    except Exception as exc:  # noqa: BLE001
        # Do not fail pipeline on upload error
        logger.exception("R2 upload failed: %s", exc)
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
    logger.info(
        "Video generation requested: name=%s, interval=%.2f, upload_to_r2=%s, output_basename=%s",
        payload.name,
        payload.group_interval_seconds,
        payload.upload_to_r2,
        payload.output_basename,
    )

    # 1) Storyline
    storyline = await _generate_storyline(payload.system_prompt, payload.biography)
    logger.info("Storyline generated (length=%d)", len(storyline))

    # 2) ElevenLabs TTS
    audio_name = payload.output_basename or datetime.utcnow().strftime("%Y%m%d%H%M%S")
    audio_path = await _elevenlabs_tts(storyline, filename=f"{audio_name}.mp3")
    logger.info("TTS audio created: %s", audio_path)

    # 3) Whisper transcription (word timestamps)
    whisper_json = await _openai_whisper_transcribe(audio_path)
    whisper_path = WHISPER_DIR / "transcript.json"
    whisper_path.write_text(json.dumps(whisper_json, ensure_ascii=False), encoding="utf-8")
    logger.info("Whisper transcript saved to %s", whisper_path)

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
    logger.info("Normalized words count=%d", len(words))

    groups = _group_words_by_interval(words, payload.group_interval_seconds)
    logger.info("Computed %d groups for image generation", len(groups))

    # 4) Generate images in parallel: Deepseek prompt -> Together FLUX
    async def gen_image(idx: int, text: str) -> Path:
        prompt = await _deepseek_image_prompt(payload.name, text, payload.image_context or "")
        img_bytes = await _together_flux_image(prompt)
        return _write_image(idx, img_bytes, audio_name)

    # To respect Together limits (default 1 QPS), generate sequentially by default.
    # You can opt-in to limited concurrency via TOGETHER_CONCURRENCY env var.
    concurrency_env = os.getenv("TOGETHER_CONCURRENCY", "1")
    try:
        max_concurrency = max(int(concurrency_env), 1)
    except Exception:
        max_concurrency = 1

    texts = [g["text"] for g in groups if g["text"]]
    logger.info("Starting generation of %d images with concurrency=%d", len(texts), max_concurrency)

    image_paths: List[Path] = []
    if max_concurrency <= 1:
        # Sequential to avoid rate limits
        for i, text in enumerate(texts, start=1):
            image_paths.append(await gen_image(i, text))
    else:
        # Bounded concurrency
        sem = asyncio.Semaphore(max_concurrency)

        async def worker(i: int, t: str) -> Path:
            async with sem:
                return await gen_image(i, t)

        tasks = [worker(i, t) for i, t in enumerate(texts, start=1)]
        image_paths = await asyncio.gather(*tasks)
    logger.info("Generated %d images", len(image_paths))

    # 5) Create video
    output_path = OUTPUT_DIR / f"{audio_name}.mp4"

    def _render_video() -> None:
        logger.info("Rendering video to %s", output_path)
        create_video_from_assets(
            images=image_paths,
            audio_file=str(audio_path),
            transcript_path=str(whisper_path),
            output_path=str(output_path),
            fps=60,
            background_audio_dir=str(BACKGROUND_AUDIO_DIR),
            background_volume=0.3,
        )

    # MoviePy is synchronous/CPU-bound; run in threadpool to keep event loop responsive
    await asyncio.to_thread(_render_video)
    logger.info("Video rendering completed: %s", output_path)

    # 6) Optional upload to R2
    output_url: Optional[str] = None
    if payload.upload_to_r2:
        key = f"{audio_name}.mp4"
        output_url = _maybe_upload_to_r2(output_path, key)
        if output_url:
            logger.info("Upload to R2 successful: %s", output_url)
        else:
            logger.warning("Upload to R2 skipped or failed")

    # Optionally: Title/Caption generation (kept minimal; can be extended)
    title = None
    caption = None

    response = GenerateVideoResponse(
        title=title,
        caption=caption,
        output_path=str(output_path),
        output_url=output_url,
        images_generated=len(image_paths),
        audio_seconds=_seconds_from_whisper(whisper_json),
    )
    logger.info(
        "Responding: images_generated=%d, audio_seconds=%.2f, output=%s",
        response.images_generated,
        response.audio_seconds,
        response.output_path,
    )
    return response


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("src.service:app", host="0.0.0.0", port=8000, reload=False)


