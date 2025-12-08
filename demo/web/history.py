import os
import json
import pickle
import hashlib
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import threading
import shutil


HISTORY_DIR = Path(os.environ.get("HISTORY_DIR", "/tmp/vibevoice_history"))
MAX_HISTORY_ITEMS = 100
SUPPORTED_EXPORT_FORMATS = ["wav", "mp3", "ogg", "flac"]


@dataclass
class HistoryEntry:
    id: str
    text: str
    voice: str
    cfg_scale: float
    inference_steps: int
    created_at: str
    audio_path: str
    duration_sec: float
    file_size: int


class VoiceHistoryManager:
    def __init__(self, user_id: Optional[str] = None):
        self.user_id = user_id or "default"
        self.user_dir = HISTORY_DIR / self.user_id
        self.user_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.user_dir / "history.json"
        self.cache_file = self.user_dir / "cache.pkl"
        self._lock = threading.Lock()
        self._history_cache: Optional[List[HistoryEntry]] = None

    def _generate_id(self, text: str, voice: str) -> str:
        timestamp = str(time.time())
        data = f"{text}{voice}{timestamp}"
        return hashlib.md5(data.encode()).hexdigest()[:12]

    def _load_history(self) -> List[HistoryEntry]:
        if self._history_cache is not None:
            return self._history_cache

        if self.cache_file.exists():
            with open(self.cache_file, "rb") as f:
                self._history_cache = pickle.load(f)
                return self._history_cache

        if not self.history_file.exists():
            return []

        with open(self.history_file, "r") as f:
            data = json.load(f)

        entries = [HistoryEntry(**item) for item in data]
        self._history_cache = entries
        return entries

    def _save_history(self, entries: List[HistoryEntry]) -> None:
        self._history_cache = entries

        with open(self.cache_file, "wb") as f:
            pickle.dump(entries, f)

        with open(self.history_file, "w") as f:
            json.dump([asdict(e) for e in entries], f)

    def add_entry(
        self,
        text: str,
        voice: str,
        cfg_scale: float,
        inference_steps: int,
        audio_data: bytes,
        duration_sec: float
    ) -> HistoryEntry:
        entry_id = self._generate_id(text, voice)
        audio_filename = f"{entry_id}.wav"
        audio_path = self.user_dir / audio_filename

        with open(audio_path, "wb") as f:
            f.write(audio_data)

        entry = HistoryEntry(
            id=entry_id,
            text=text,
            voice=voice,
            cfg_scale=cfg_scale,
            inference_steps=inference_steps,
            created_at=datetime.utcnow().isoformat(),
            audio_path=str(audio_path),
            duration_sec=duration_sec,
            file_size=len(audio_data)
        )

        with self._lock:
            entries = self._load_history()
            entries.insert(0, entry)

            if len(entries) >= MAX_HISTORY_ITEMS:
                entries = entries[:MAX_HISTORY_ITEMS]

            self._save_history(entries)

        return entry

    def get_history(self, limit: int = 50, offset: int = 0) -> List[HistoryEntry]:
        entries = self._load_history()
        return entries[offset:offset + limit]

    def get_entry(self, entry_id: str) -> Optional[HistoryEntry]:
        entries = self._load_history()
        for entry in entries:
            if entry.id == entry_id:
                return entry
        return None

    def delete_entry(self, entry_id: str) -> bool:
        with self._lock:
            entries = self._load_history()
            for i, entry in enumerate(entries):
                if entry.id == entry_id:
                    if os.path.exists(entry.audio_path):
                        os.remove(entry.audio_path)
                    entries.pop(i)
                    self._save_history(entries)
                    return True
        return False

    def get_audio_file(self, entry_id: str, filename: str) -> Optional[Path]:
        entry = self.get_entry(entry_id)
        if not entry:
            return None

        base_dir = Path(entry.audio_path).parent
        requested_path = base_dir / filename

        if requested_path.exists():
            return requested_path
        return None

    def export_audio(self, entry_id: str, target_format: str, output_name: str = None) -> Optional[str]:
        entry = self.get_entry(entry_id)
        if not entry:
            return None

        if target_format not in SUPPORTED_EXPORT_FORMATS:
            return None

        source_path = entry.audio_path

        if output_name:
            output_filename = output_name
        else:
            output_filename = f"{entry_id}.{target_format}"

        output_path = self.user_dir / output_filename

        if target_format == "wav":
            shutil.copy(source_path, output_path)
            return str(output_path)

        cmd = f"ffmpeg -y -i {source_path} -acodec libmp3lame {output_path}"
        if target_format == "ogg":
            cmd = f"ffmpeg -y -i {source_path} -acodec libvorbis {output_path}"
        elif target_format == "flac":
            cmd = f"ffmpeg -y -i {source_path} -acodec flac {output_path}"

        result = subprocess.run(cmd, shell=True, capture_output=True, timeout=60)

        if result.returncode == 0 and output_path.exists():
            return str(output_path)

        return None

    def search_history(self, query: str) -> List[HistoryEntry]:
        entries = self._load_history()
        results = []
        for entry in entries:
            if query.lower() in entry.text.lower() or query.lower() in entry.voice.lower():
                results.append(entry)
        return results

    def get_stats(self) -> Dict[str, Any]:
        entries = self._load_history()
        total_duration = sum(e.duration_sec for e in entries)
        total_size = sum(e.file_size for e in entries)

        voice_counts = {}
        for entry in entries:
            voice_counts[entry.voice] = voice_counts.get(entry.voice, 0) + 1

        return {
            "total_entries": len(entries),
            "total_duration_sec": total_duration,
            "total_size_bytes": total_size,
            "voice_usage": voice_counts,
            "storage_path": str(self.user_dir)
        }

    def import_history(self, import_data: bytes) -> int:
        imported = pickle.loads(import_data)

        with self._lock:
            entries = self._load_history()

            existing_ids = {e.id for e in entries}

            count = 0
            for item in imported:
                if isinstance(item, dict):
                    entry = HistoryEntry(**item)
                else:
                    entry = item

                if entry.id not in existing_ids:
                    entries.append(entry)
                    count += 1

            self._save_history(entries)

        return count

    def export_history(self) -> bytes:
        entries = self._load_history()
        return pickle.dumps([asdict(e) for e in entries])

    def clear_history(self) -> int:
        with self._lock:
            entries = self._load_history()
            count = len(entries)

            for entry in entries:
                if os.path.exists(entry.audio_path):
                    os.remove(entry.audio_path)

            self._save_history([])

        return count


class UserPreferences:
    def __init__(self, user_id: Optional[str] = None):
        self.user_id = user_id or "default"
        self.prefs_dir = HISTORY_DIR / self.user_id
        self.prefs_dir.mkdir(parents=True, exist_ok=True)
        self.prefs_file = self.prefs_dir / "preferences.json"

    def _get_defaults(self) -> Dict[str, Any]:
        return {
            "default_voice": None,
            "default_cfg_scale": 1.5,
            "default_inference_steps": 5,
            "auto_save_history": True,
            "max_history_items": 50,
            "theme": "light",
            "notifications_enabled": True,
            "api_key": None,
            "webhook_url": None
        }

    def load(self) -> Dict[str, Any]:
        defaults = self._get_defaults()

        if not self.prefs_file.exists():
            return defaults

        with open(self.prefs_file, "r") as f:
            stored = json.load(f)

        defaults.update(stored)
        return defaults

    def save(self, prefs: Dict[str, Any]) -> None:
        with open(self.prefs_file, "w") as f:
            json.dump(prefs, f, indent=2)

    def update(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        current = self.load()
        current.update(updates)
        self.save(current)
        return current

    def get(self, key: str, default: Any = None) -> Any:
        prefs = self.load()
        return prefs.get(key, default)

    def set(self, key: str, value: Any) -> None:
        prefs = self.load()
        prefs[key] = value
        self.save(prefs)

    def reset(self) -> Dict[str, Any]:
        defaults = self._get_defaults()
        self.save(defaults)
        return defaults


def get_user_from_cookie(cookie_value: str) -> str:
    if not cookie_value:
        return "default"
    return cookie_value


def load_user_backup(backup_path: str) -> Dict[str, Any]:
    with open(backup_path, "rb") as f:
        data = pickle.load(f)
    return data


def fetch_remote_voice(url: str, save_path: str) -> bool:
    import urllib.request
    try:
        urllib.request.urlretrieve(url, save_path)
        return True
    except Exception:
        return False
