"""
app/inference/cache.py — LRU model cache with lazy loading and hot reload.

Wraps the existing ``app.models.load_model`` module with:

  - An LRU eviction policy (configurable capacity).
  - Per-model load timestamps and hit counters.
  - A hot-reload mechanism that forces a fresh load from disk.
  - Thread-safe operations via ``threading.RLock``.
  - Memory cleanup via explicit eviction.

The module exposes a process-wide singleton (``_cache``) and a set of
module-level convenience functions so callers never need to instantiate
``ModelCache`` directly.

Usage
-----
    from app.inference.cache import get_model, reload_model, cache_stats

    model = get_model("efficientnet")     # loads & caches
    model = get_model("efficientnet")     # cache hit
    reload_model("efficientnet")          # hot reload from disk
    stats = cache_stats()                 # hit/miss/size info
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.core.logging import logger
from app.models.load_model import (
    _resolve_model_path,
    clear_model_cache,
    get_model_info,
    is_model_available,
    load_model,
)


# ─── Cache entry ──────────────────────────────────────────────────────────────

class _CacheEntry:
    """Internal wrapper around a loaded model adapter."""

    __slots__ = (
        "model", "model_name", "loaded_at", "last_accessed_at",
        "hit_count", "load_duration_ms", "model_info",
    )

    def __init__(
        self,
        model: Any,
        model_name: str,
        load_duration_ms: float,
        model_info: Dict[str, Any],
    ) -> None:
        self.model            = model
        self.model_name       = model_name
        self.loaded_at        = datetime.now(timezone.utc).isoformat()
        self.last_accessed_at = self.loaded_at
        self.hit_count        = 0
        self.load_duration_ms = round(load_duration_ms, 2)
        self.model_info       = model_info

    def touch(self) -> None:
        self.last_accessed_at = datetime.now(timezone.utc).isoformat()
        self.hit_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name":       self.model_name,
            "loaded_at":        self.loaded_at,
            "last_accessed_at": self.last_accessed_at,
            "hit_count":        self.hit_count,
            "load_duration_ms": self.load_duration_ms,
            "total_params":     self.model.count_params(),
            "model_version":    self.model_info.get("saved_at"),
        }


# ─── ModelCache ───────────────────────────────────────────────────────────────

class ModelCache:
    """
    Thread-safe LRU cache for models.

    Parameters
    ----------
    capacity : int
        Maximum number of models held simultaneously.
        When exceeded the least-recently-used model is evicted.
    """

    def __init__(self, capacity: int = 4) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._lock     = threading.RLock()
        # OrderedDict preserves insertion order; we move-to-end on access
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._total_hits   = 0
        self._total_misses = 0

    # ── Core operations ───────────────────────────────────────────────────────

    def get(self, model_name: str) -> Optional[Any]:
        """
        Return a cached model or None (does NOT auto-load).

        Updates LRU order and hit counter on success.
        """
        name = model_name.lower()
        with self._lock:
            entry = self._store.get(name)
            if entry is None:
                self._total_misses += 1
                return None
            # Move to end = most-recently-used
            self._store.move_to_end(name)
            entry.touch()
            self._total_hits += 1
            return entry.model

    def load(self, model_name: str) -> Any:
        """
        Load *model_name* from disk, insert into the cache, and return it.

        Evicts the LRU entry when capacity is reached.

        Raises
        ------
        FileNotFoundError
            When no saved weights exist for *model_name*.
        RuntimeError
            When the model fails to deserialise.
        """
        name = model_name.lower()
        model_path = _resolve_model_path(name)
        logger.info(f"[Cache] Loading '{name}' from {model_path} …")

        t0 = time.perf_counter()
        try:
            clear_model_cache(name)
            model = load_model(name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model '{name}' from {model_path}: {exc}"
            ) from exc
        load_ms = (time.perf_counter() - t0) * 1000

        info = get_model_info(name)
        entry = _CacheEntry(model, name, load_ms, info)

        with self._lock:
            # Evict LRU if at capacity (and model not already in cache)
            if name not in self._store and len(self._store) >= self._capacity:
                evicted_name, _ = self._store.popitem(last=False)
                logger.info(f"[Cache] Evicted '{evicted_name}' (LRU, capacity={self._capacity})")

            self._store[name] = entry
            self._store.move_to_end(name)

        logger.info(
            f"[Cache] '{name}' loaded and cached | "
            f"load_ms={load_ms:.1f} params={model.count_params():,}"
        )
        return model

    def get_or_load(self, model_name: str) -> Any:
        """
        Return cached model, loading from disk if not present.

        This is the primary entry point for inference code.
        """
        model = self.get(model_name)
        if model is not None:
            logger.debug(f"[Cache] Hit for '{model_name.lower()}'")
            return model
        return self.load(model_name)

    def reload(self, model_name: str) -> Any:
        """
        Force a fresh load from disk, replacing any cached entry.

        Use this after a model has been re-trained (hot reload).

        Returns
        -------
        Any
            Freshly loaded model.
        """
        name = model_name.lower()
        with self._lock:
            if name in self._store:
                del self._store[name]
                logger.info(f"[Cache] Evicted '{name}' for hot reload.")
        return self.load(name)

    def evict(self, model_name: str) -> bool:
        """
        Remove a model from the cache without reloading.

        Returns True when the model was present.
        """
        name = model_name.lower()
        with self._lock:
            if name in self._store:
                del self._store[name]
                logger.info(f"[Cache] '{name}' evicted.")
                return True
        return False

    def clear(self) -> int:
        """Evict all models. Returns the number removed."""
        with self._lock:
            count = len(self._store)
            self._store.clear()
        logger.info(f"[Cache] Cleared {count} model(s).")
        return count

    # ── Introspection ─────────────────────────────────────────────────────────

    def is_cached(self, model_name: str) -> bool:
        with self._lock:
            return model_name.lower() in self._store

    def cached_names(self) -> List[str]:
        with self._lock:
            return list(self._store.keys())

    def entry_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Return metadata for a cached model, or None."""
        with self._lock:
            entry = self._store.get(model_name.lower())
            return entry.to_dict() if entry else None

    def stats(self) -> Dict[str, Any]:
        """Return aggregate cache statistics."""
        with self._lock:
            total_requests = self._total_hits + self._total_misses
            hit_rate = (
                round(self._total_hits / total_requests, 4)
                if total_requests > 0
                else 0.0
            )
            return {
                "capacity":       self._capacity,
                "size":           len(self._store),
                "cached_models":  list(self._store.keys()),
                "total_hits":     self._total_hits,
                "total_misses":   self._total_misses,
                "hit_rate":       hit_rate,
                "entries":        [e.to_dict() for e in self._store.values()],
            }

    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        Scan the saved_models directory and return availability info for
        all known architectures.

        Returns
        -------
        list[dict]
            Each entry: {name, available, cached, model_info (if available)}
        """
        from app.core.config import settings
        architectures = ("mambavision", "cnn", "vgg16", "resnet50", "efficientnet")
        results = []
        for arch in architectures:
            available = is_model_available(arch)
            cached    = self.is_cached(arch)
            info      = get_model_info(arch) if available else {}
            results.append({
                "name":          arch,
                "available":     available,
                "cached":        cached,
                "model_version": info.get("saved_at"),
                "total_params":  info.get("total_params"),
                "model_dir":     str(settings.saved_models_dir / arch),
            })
        return results

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __contains__(self, model_name: str) -> bool:
        return self.is_cached(model_name)


# ── Process-wide singleton ─────────────────────────────────────────────────────
_cache = ModelCache(capacity=4)


# ── Module-level convenience functions ────────────────────────────────────────

def get_model(model_name: str) -> Any:
    """Load (cached) the requested model — primary inference entry point."""
    return _cache.get_or_load(model_name)


def reload_model(model_name: str) -> Any:
    """Hot-reload: evict from cache and reload from disk."""
    return _cache.reload(model_name)


def evict_model(model_name: str) -> bool:
    """Remove a model from the cache without reloading."""
    return _cache.evict(model_name)


def clear_cache() -> int:
    """Clear all cached models. Returns count removed."""
    return _cache.clear()


def cache_stats() -> Dict[str, Any]:
    """Return aggregate cache statistics."""
    return _cache.stats()


def list_available_models() -> List[Dict[str, Any]]:
    """Scan saved_models/ and return per-architecture availability."""
    return _cache.list_available_models()
