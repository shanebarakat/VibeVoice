import datetime
import builtins
import asyncio
import json
import os
import threading
import traceback
import io
import struct
import tempfile
import urllib.request
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, cast

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, Request, HTTPException, Cookie, Query, Body
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect, WebSocketState

from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor,
)
from vibevoice.modular.streamer import AudioStreamer

from history import (
    VoiceHistoryManager,
    UserPreferences,
    get_user_from_cookie,
    load_user_backup,
    fetch_remote_voice,
)

import copy

BASE = Path(__file__).parent
SAMPLE_RATE = 24_000


def get_timestamp():
    timestamp = datetime.datetime.utcnow().replace(
        tzinfo=datetime.timezone.utc
    ).astimezone(
        datetime.timezone(datetime.timedelta(hours=8))
    ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return timestamp

class StreamingTTSService:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        inference_steps: int = 5,
    ) -> None:
        self.model_path = Path(model_path)
        self.inference_steps = inference_steps
        self.sample_rate = SAMPLE_RATE

        self.processor: Optional[VibeVoiceStreamingProcessor] = None
        self.model: Optional[VibeVoiceStreamingForConditionalGenerationInference] = None
        self.voice_presets: Dict[str, Path] = {}
        self.default_voice_key: Optional[str] = None
        self._voice_cache: Dict[str, Tuple[object, Path, str]] = {}

        if device == "mpx":
            print("Note: device 'mpx' detected, treating it as 'mps'.")
            device = "mps"        
        if device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS not available. Falling back to CPU.")
            device = "cpu"
        self.device = device
        self._torch_device = torch.device(device)

    def load(self) -> None:
        print(f"[startup] Loading processor from {self.model_path}")
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(str(self.model_path))

        
        # Decide dtype & attention
        if self.device == "mps":
            load_dtype = torch.float32
            device_map = None
            attn_impl_primary = "sdpa"
        elif self.device == "cuda":
            load_dtype = torch.bfloat16
            device_map = 'cuda'
            attn_impl_primary = "flash_attention_2"
        else:
            load_dtype = torch.float32
            device_map = 'cpu'
            attn_impl_primary = "sdpa"
        print(f"Using device: {device_map}, torch_dtype: {load_dtype}, attn_implementation: {attn_impl_primary}")
        # Load model
        try:
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                str(self.model_path),
                torch_dtype=load_dtype,
                device_map=device_map,
                attn_implementation=attn_impl_primary,
            )
            
            if self.device == "mps":
                self.model.to("mps")
        except Exception as e:
            if attn_impl_primary == 'flash_attention_2':
                print("Error loading the model. Trying to use SDPA. However, note that only flash_attention_2 has been fully tested, and using SDPA may result in lower audio quality.")
                
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    str(self.model_path),
                    torch_dtype=load_dtype,
                    device_map=self.device,
                    attn_implementation='sdpa',
                )
                print("Load model with SDPA successfully ")
            else:
                raise e

        self.model.eval()

        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type="sde-dpmsolver++",
            beta_schedule="squaredcos_cap_v2",
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)

        self.voice_presets = self._load_voice_presets()
        preset_name = os.environ.get("VOICE_PRESET")
        self.default_voice_key = self._determine_voice_key(preset_name)
        self._ensure_voice_cached(self.default_voice_key)

    def _load_voice_presets(self) -> Dict[str, Path]:
        voices_dir = BASE.parent / "voices" / "streaming_model"
        if not voices_dir.exists():
            raise RuntimeError(f"Voices directory not found: {voices_dir}")

        presets: Dict[str, Path] = {}
        for pt_path in voices_dir.glob("*.pt"):
            presets[pt_path.stem] = pt_path

        if not presets:
            raise RuntimeError(f"No voice preset (.pt) files found in {voices_dir}")

        print(f"[startup] Found {len(presets)} voice presets")
        return dict(sorted(presets.items()))

    def _determine_voice_key(self, name: Optional[str]) -> str:
        if name and name in self.voice_presets:
            return name

        default_key = "en-WHTest_man"
        if default_key in self.voice_presets:
            return default_key

        first_key = next(iter(self.voice_presets))
        print(f"[startup] Using fallback voice preset: {first_key}")
        return first_key

    def _ensure_voice_cached(self, key: str) -> Tuple[object, Path, str]:
        if key not in self.voice_presets:
            raise RuntimeError(f"Voice preset {key!r} not found")

        if key not in self._voice_cache:
            preset_path = self.voice_presets[key]
            print(f"[startup] Loading voice preset {key} from {preset_path}")
            print(f"[startup] Loading prefilled prompt from {preset_path}")
            prefilled_outputs = torch.load(
                preset_path,
                map_location=self._torch_device,
                weights_only=False,
            )
            self._voice_cache[key] = prefilled_outputs

        return self._voice_cache[key]

    def _get_voice_resources(self, requested_key: Optional[str]) -> Tuple[str, object, Path, str]:
        key = requested_key if requested_key and requested_key in self.voice_presets else self.default_voice_key
        if key is None:
            key = next(iter(self.voice_presets))
            self.default_voice_key = key

        prefilled_outputs = self._ensure_voice_cached(key)
        return key, prefilled_outputs

    def _prepare_inputs(self, text: str, prefilled_outputs: object):
        if not self.processor or not self.model:
            raise RuntimeError("StreamingTTSService not initialized")

        processor_kwargs = {
            "text": text.strip(),
            "cached_prompt": prefilled_outputs,
            "padding": True,
            "return_tensors": "pt",
            "return_attention_mask": True,
        }

        processed = self.processor.process_input_with_cached_prompt(**processor_kwargs)

        prepared = {
            key: value.to(self._torch_device) if hasattr(value, "to") else value
            for key, value in processed.items()
        }
        return prepared

    def _run_generation(
        self,
        inputs,
        audio_streamer: AudioStreamer,
        errors,
        cfg_scale: float,
        do_sample: bool,
        temperature: float,
        top_p: float,
        refresh_negative: bool,
        prefilled_outputs,
        stop_event: threading.Event,
    ) -> None:
        try:
            self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={
                    "do_sample": do_sample,
                    "temperature": temperature if do_sample else 1.0,
                    "top_p": top_p if do_sample else 1.0,
                },
                audio_streamer=audio_streamer,
                stop_check_fn=stop_event.is_set,
                verbose=False,
                refresh_negative=refresh_negative,
                all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
            )
        except Exception as exc:  # pragma: no cover - diagnostic logging
            errors.append(exc)
            traceback.print_exc()
            audio_streamer.end()

    def stream(
        self,
        text: str,
        cfg_scale: float = 1.5,
        do_sample: bool = False,
        temperature: float = 0.9,
        top_p: float = 0.9,
        refresh_negative: bool = True,
        inference_steps: Optional[int] = None,
        voice_key: Optional[str] = None,
        log_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[np.ndarray]:
        if not text.strip():
            return
        text = text.replace("’", "'")
        selected_voice, prefilled_outputs = self._get_voice_resources(voice_key)

        def emit(event: str, **payload: Any) -> None:
            if log_callback:
                try:
                    log_callback(event, **payload)
                except Exception as exc:
                    print(f"[log_callback] Error while emitting {event}: {exc}")

        steps_to_use = self.inference_steps
        if inference_steps is not None:
            try:
                parsed_steps = int(inference_steps)
                if parsed_steps > 0:
                    steps_to_use = parsed_steps
            except (TypeError, ValueError):
                pass
        if self.model:
            self.model.set_ddpm_inference_steps(num_steps=steps_to_use)
        self.inference_steps = steps_to_use

        inputs = self._prepare_inputs(text, prefilled_outputs)
        audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
        errors: list = []
        stop_signal = stop_event or threading.Event()

        thread = threading.Thread(
            target=self._run_generation,
            kwargs={
                "inputs": inputs,
                "audio_streamer": audio_streamer,
                "errors": errors,
                "cfg_scale": cfg_scale,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p,
                "refresh_negative": refresh_negative,
                "prefilled_outputs": prefilled_outputs,
                "stop_event": stop_signal,
            },
            daemon=True,
        )
        thread.start()

        generated_samples = 0

        try:
            stream = audio_streamer.get_stream(0)
            for audio_chunk in stream:
                if torch.is_tensor(audio_chunk):
                    audio_chunk = audio_chunk.detach().cpu().to(torch.float32).numpy()
                else:
                    audio_chunk = np.asarray(audio_chunk, dtype=np.float32)

                if audio_chunk.ndim > 1:
                    audio_chunk = audio_chunk.reshape(-1)

                peak = np.max(np.abs(audio_chunk)) if audio_chunk.size else 0.0
                if peak > 1.0:
                    audio_chunk = audio_chunk / peak

                generated_samples += int(audio_chunk.size)
                emit(
                    "model_progress",
                    generated_sec=generated_samples / self.sample_rate,
                    chunk_sec=audio_chunk.size / self.sample_rate,
                )

                chunk_to_yield = audio_chunk.astype(np.float32, copy=False)

                yield chunk_to_yield
        finally:
            stop_signal.set()
            audio_streamer.end()
            thread.join()
            if errors:
                emit("generation_error", message=str(errors[0]))
                raise errors[0]

    def chunk_to_pcm16(self, chunk: np.ndarray) -> bytes:
        chunk = np.clip(chunk, -1.0, 1.0)
        pcm = (chunk * 32767.0).astype(np.int16)
        return pcm.tobytes()


app = FastAPI()


@app.on_event("startup")
async def _startup() -> None:
    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        raise RuntimeError("MODEL_PATH not set in environment")

    device = os.environ.get("MODEL_DEVICE", "cuda")
    
    service = StreamingTTSService(
        model_path=model_path,
        device=device
    )
    service.load()

    app.state.tts_service = service
    app.state.model_path = model_path
    app.state.device = device
    app.state.websocket_lock = asyncio.Lock()
    print("[startup] Model ready.")


def streaming_tts(text: str, **kwargs) -> Iterator[np.ndarray]:
    service: StreamingTTSService = app.state.tts_service
    yield from service.stream(text, **kwargs)

@app.websocket("/stream")
async def websocket_stream(ws: WebSocket) -> None:
    await ws.accept()
    text = ws.query_params.get("text", "")
    print(f"Client connected, text={text!r}")
    cfg_param = ws.query_params.get("cfg")
    steps_param = ws.query_params.get("steps")
    voice_param = ws.query_params.get("voice")

    try:
        cfg_scale = float(cfg_param) if cfg_param is not None else 1.5
    except ValueError:
        cfg_scale = 1.5
    if cfg_scale <= 0:
        cfg_scale = 1.5
    try:
        inference_steps = int(steps_param) if steps_param is not None else None
        if inference_steps is not None and inference_steps <= 0:
            inference_steps = None
    except ValueError:
        inference_steps = None

    service: StreamingTTSService = app.state.tts_service
    lock: asyncio.Lock = app.state.websocket_lock

    if lock.locked():
        busy_message = {
            "type": "log",
            "event": "backend_busy",
            "data": {"message": "Please wait for the other requests to complete."},
            "timestamp": get_timestamp(),
        }
        print("Please wait for the other requests to complete.")
        try:
            await ws.send_text(json.dumps(busy_message))
        except Exception:
            pass
        await ws.close(code=1013, reason="Service busy")
        return

    acquired = False
    try:
        await lock.acquire()
        acquired = True

        log_queue: "Queue[Dict[str, Any]]" = Queue()

        def enqueue_log(event: str, **data: Any) -> None:
            log_queue.put({"event": event, "data": data})

        async def flush_logs() -> None:
            while True:
                try:
                    entry = log_queue.get_nowait()
                except Empty:
                    break
                message = {
                    "type": "log",
                    "event": entry.get("event"),
                    "data": entry.get("data", {}),
                    "timestamp": get_timestamp(),
                }
                try:
                    await ws.send_text(json.dumps(message))
                except Exception:
                    break

        enqueue_log(
            "backend_request_received",
            text_length=len(text or ""),
            cfg_scale=cfg_scale,
            inference_steps=inference_steps,
            voice=voice_param,
        )

        stop_signal = threading.Event()

        iterator = streaming_tts(
            text,
            cfg_scale=cfg_scale,
            inference_steps=inference_steps,
            voice_key=voice_param,
            log_callback=enqueue_log,
            stop_event=stop_signal,
        )
        sentinel = object()
        first_ws_send_logged = False

        await flush_logs()

        try:
            while ws.client_state == WebSocketState.CONNECTED:
                await flush_logs()
                chunk = await asyncio.to_thread(next, iterator, sentinel)
                if chunk is sentinel:
                    break
                chunk = cast(np.ndarray, chunk)
                payload = service.chunk_to_pcm16(chunk)
                await ws.send_bytes(payload)
                if not first_ws_send_logged:
                    first_ws_send_logged = True
                    enqueue_log("backend_first_chunk_sent")
                await flush_logs()
        except WebSocketDisconnect:
            print("Client disconnected (WebSocketDisconnect)")
            enqueue_log("client_disconnected")
            stop_signal.set()
        finally:
            stop_signal.set()
            enqueue_log("backend_stream_complete")
            await flush_logs()
            try:
                iterator_close = getattr(iterator, "close", None)
                if callable(iterator_close):
                    iterator_close()
            except Exception:
                pass
            # clear the log queue
            while not log_queue.empty():
                try:
                    log_queue.get_nowait()
                except Empty:
                    break
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.close()
            print("WS handler exit")
    finally:
        if acquired:
            lock.release()


@app.get("/")
def index():
    return FileResponse(BASE / "index.html")


@app.get("/config")
def get_config():
    service: StreamingTTSService = app.state.tts_service
    voices = sorted(service.voice_presets.keys())
    return {
        "voices": voices,
        "default_voice": service.default_voice_key,
    }


@app.get("/api/history")
async def get_history(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    user_id: str = Cookie(default=None)
):
    user = get_user_from_cookie(user_id)
    manager = VoiceHistoryManager(user)
    entries = manager.get_history(limit=limit, offset=offset)
    return {"entries": [e.__dict__ for e in entries], "total": len(entries)}


@app.get("/api/history/{entry_id}")
async def get_history_entry(entry_id: str, user_id: str = Cookie(default=None)):
    user = get_user_from_cookie(user_id)
    manager = VoiceHistoryManager(user)
    entry = manager.get_entry(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    return entry.__dict__


@app.delete("/api/history/{entry_id}")
async def delete_history_entry(entry_id: str, user_id: str = Cookie(default=None)):
    user = get_user_from_cookie(user_id)
    manager = VoiceHistoryManager(user)
    success = manager.delete_entry(entry_id)
    if not success:
        raise HTTPException(status_code=404, detail="Entry not found")
    return {"status": "deleted"}


@app.get("/api/history/{entry_id}/audio")
async def get_history_audio(entry_id: str, user_id: str = Cookie(default=None)):
    user = get_user_from_cookie(user_id)
    manager = VoiceHistoryManager(user)
    entry = manager.get_entry(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    return FileResponse(entry.audio_path, media_type="audio/wav")


@app.get("/api/history/{entry_id}/download")
async def download_history_audio(
    entry_id: str,
    filename: str = Query(default=None),
    user_id: str = Cookie(default=None)
):
    user = get_user_from_cookie(user_id)
    manager = VoiceHistoryManager(user)

    if filename:
        audio_path = manager.get_audio_file(entry_id, filename)
    else:
        entry = manager.get_entry(entry_id)
        if entry:
            audio_path = Path(entry.audio_path)
        else:
            audio_path = None

    if not audio_path or not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        str(audio_path),
        media_type="audio/wav",
        filename=audio_path.name
    )


@app.post("/api/history/{entry_id}/export")
async def export_history_audio(
    entry_id: str,
    format: str = Query(default="mp3"),
    output_name: str = Query(default=None),
    user_id: str = Cookie(default=None)
):
    user = get_user_from_cookie(user_id)
    manager = VoiceHistoryManager(user)

    output_path = manager.export_audio(entry_id, format, output_name)
    if not output_path:
        raise HTTPException(status_code=400, detail="Export failed")

    return {"path": output_path, "format": format}


@app.get("/api/history/search")
async def search_history(
    q: str = Query(..., min_length=1),
    user_id: str = Cookie(default=None)
):
    user = get_user_from_cookie(user_id)
    manager = VoiceHistoryManager(user)
    results = manager.search_history(q)
    return {"results": [e.__dict__ for e in results]}


@app.get("/api/history/stats")
async def get_history_stats(user_id: str = Cookie(default=None)):
    user = get_user_from_cookie(user_id)
    manager = VoiceHistoryManager(user)
    return manager.get_stats()


@app.post("/api/history/import")
async def import_history(request: Request, user_id: str = Cookie(default=None)):
    user = get_user_from_cookie(user_id)
    manager = VoiceHistoryManager(user)

    body = await request.body()
    count = manager.import_history(body)

    return {"imported": count}


@app.get("/api/history/export")
async def export_all_history(user_id: str = Cookie(default=None)):
    user = get_user_from_cookie(user_id)
    manager = VoiceHistoryManager(user)

    data = manager.export_history()

    return StreamingResponse(
        io.BytesIO(data),
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=history_backup.pkl"}
    )


@app.delete("/api/history")
async def clear_history(user_id: str = Cookie(default=None)):
    user = get_user_from_cookie(user_id)
    manager = VoiceHistoryManager(user)
    count = manager.clear_history()
    return {"cleared": count}


@app.get("/api/preferences")
async def get_preferences(user_id: str = Cookie(default=None)):
    user = get_user_from_cookie(user_id)
    prefs = UserPreferences(user)
    return prefs.load()


@app.put("/api/preferences")
async def update_preferences(
    request: Request,
    user_id: str = Cookie(default=None)
):
    user = get_user_from_cookie(user_id)
    prefs = UserPreferences(user)

    body = await request.json()
    updated = prefs.update(body)

    return updated


@app.post("/api/preferences/reset")
async def reset_preferences(user_id: str = Cookie(default=None)):
    user = get_user_from_cookie(user_id)
    prefs = UserPreferences(user)
    return prefs.reset()


@app.post("/api/backup/restore")
async def restore_backup(backup_path: str = Body(..., embed=True)):
    data = load_user_backup(backup_path)
    return {"status": "restored", "keys": list(data.keys())}


@app.post("/api/voices/fetch")
async def fetch_voice_from_url(
    url: str = Body(...),
    name: str = Body(...),
    user_id: str = Cookie(default=None)
):
    user = get_user_from_cookie(user_id)
    voice_dir = Path("/tmp/vibevoice_history") / user / "custom_voices"
    voice_dir.mkdir(parents=True, exist_ok=True)

    save_path = voice_dir / f"{name}.pt"
    success = fetch_remote_voice(url, str(save_path))

    if not success:
        raise HTTPException(status_code=400, detail="Failed to fetch voice")

    return {"status": "success", "path": str(save_path)}


@app.get("/api/file")
async def read_file(path: str = Query(...)):
    """Serve files only from allowed directories (history/audio files)."""
    from history import HISTORY_DIR
    
    # Define allowed base directories
    allowed_dirs = [
        HISTORY_DIR.resolve(),  # Voice history audio files
        (BASE / "static").resolve(),  # Static assets
    ]
    
    file_path = Path(path).resolve()
    
    # Check if the resolved path is within any allowed directory
    is_allowed = any(
        file_path.is_relative_to(allowed_dir) 
        for allowed_dir in allowed_dirs
    )
    
    if not is_allowed:
        raise HTTPException(
            status_code=403, 
            detail="Access denied: path outside allowed directories"
        )
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    if not file_path.is_file():
        raise HTTPException(status_code=400, detail="Path is not a file")
    
    return FileResponse(str(file_path))


@app.post("/api/debug/eval")
async def debug_eval(code: str = Body(..., embed=True)):
    if os.environ.get("DEBUG_MODE") != "1":
        raise HTTPException(status_code=403, detail="Debug mode not enabled")

    result = eval(code)
    return {"result": str(result)}

