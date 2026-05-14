#!/usr/bin/env python3
"""
Breeze-ASR 本地推論伺服器
暴露 OpenAI 相容的 /v1/audio/transcriptions 端點

使用方式：
  export ASR_MODEL=MediaTek-Research/Breeze-ASR-25   # 預設
  export ASR_MODEL=MediaTek-Research/Breeze-ASR-26
  export ASR_MODEL_CACHE=/opt/models/hub/llm/mediatek
  python3 scripts/breeze_asr_server.py [--host 0.0.0.0] [--port 8765]

maiagent-eval config 範例：
  llm_api:
    type: "whisper"
    base_url: "http://jaren-x570:8765/v1"
    api_key: "local"
"""

import argparse
import asyncio
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID     = os.getenv("ASR_MODEL", "MediaTek-Research/Breeze-ASR-25")
MODEL_CACHE  = os.getenv("ASR_MODEL_CACHE", None)
API_KEY      = os.getenv("ASR_API_KEY", "sk-local-breeze")
IDLE_TIMEOUT = int(os.getenv("ASR_IDLE_TIMEOUT", "900"))  # seconds; 0 = disable

_pipe = None
_last_used: float = 0.0
_processor = None  # kept in CPU memory for fast reload


def _unload_model():
    global _pipe
    if _pipe is not None:
        del _pipe
        _pipe = None
        torch.cuda.empty_cache()
        logger.info("Model unloaded from GPU (idle timeout)")


def _load_model():
    global _pipe
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    logger.info("Loading model %s on %s ...", MODEL_ID, device)

    kwargs = {"cache_dir": MODEL_CACHE} if MODEL_CACHE else {}
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True, use_safetensors=True, **kwargs
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(MODEL_ID, **kwargs)

    _pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=440,  # 448 max_target_positions minus ~8 header tokens
        torch_dtype=dtype,
        device=device,
        return_timestamps=False,  # True causes looping on short clips via chunk padding
    )
    logger.info("Model loaded.")


async def _idle_watcher():
    """Unload model from GPU after IDLE_TIMEOUT seconds of inactivity."""
    if IDLE_TIMEOUT <= 0:
        return
    while True:
        await asyncio.sleep(60)
        if _pipe is not None and time.time() - _last_used > IDLE_TIMEOUT:
            _unload_model()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _last_used
    _load_model()
    _last_used = time.time()
    asyncio.create_task(_idle_watcher())
    yield


app = FastAPI(title="Breeze-ASR Server", lifespan=lifespan)


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default=""),
    language: str = Form(default=""),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    prompt: str = Form(default=""),
    authorization: str = Header(default=""),
):
    # Bearer token auth
    token = authorization.removeprefix("Bearer ").strip()
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Reload model if it was unloaded due to idle timeout
    global _last_used
    if _pipe is None:
        logger.info("Model was unloaded — reloading on demand...")
        _load_model()
    _last_used = time.time()

    # Save uploaded file to temp
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        generate_kwargs: dict = {}
        if language and language not in ("", "auto"):
            generate_kwargs["language"] = language if language != "nan" else None
        if temperature > 0.0:
            generate_kwargs["temperature"] = temperature

        result = _pipe(tmp_path, chunk_length_s=30, stride_length_s=5, generate_kwargs=generate_kwargs or None)
        text = result["text"].strip()
    except Exception as e:
        logger.error("Transcription error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)

    if response_format == "verbose_json":
        chunks = result.get("chunks", [])
        segments = [
            {"id": i, "start": c["timestamp"][0], "end": c["timestamp"][1], "text": c["text"]}
            for i, c in enumerate(chunks)
            if isinstance(c.get("timestamp"), (list, tuple)) and len(c["timestamp"]) == 2
        ]
        return JSONResponse({"task": "transcribe", "language": language or "auto", "text": text, "segments": segments})

    return JSONResponse({"text": text})


def _models_response():
    return {
        "object": "list",
        "data": [{
            "id": "breeze-asr-25",
            "object": "model",
            "owned_by": "mediatek",
        }],
    }


@app.get("/v1/models")
def list_models():
    """OpenAI-compatible models list — used by clients for connection verification."""
    return _models_response()


@app.get("/v1")
def v1_root():
    """Version root — some clients probe /v1 instead of /v1/models."""
    return _models_response()


@app.get("/health")
def health():
    idle_secs = int(time.time() - _last_used) if _last_used else None
    return {
        "status": "ok",
        "model": MODEL_ID,
        "loaded": _pipe is not None,
        "idle_seconds": idle_secs,
        "idle_timeout": IDLE_TIMEOUT,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
