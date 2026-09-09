import json
from pathlib import Path
from typing import Any


class DocStore:
    """Persistent key-value document store for storing full raw elements (Text, HTML Tables, Base64 Images)."""

    def __init__(self, storage_path: Path | None = None):
        self.storage_path = storage_path or Path("data/processed/docstore.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._store: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self.storage_path.exists():
            try:
                content = self.storage_path.read_text(encoding="utf-8")
                self._store = json.loads(content) if content.strip() else {}
            except Exception:
                self._store = {}
        else:
            self._store = {}

    def _save(self) -> None:
        try:
            self.storage_path.write_text(
                json.dumps(self._store, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            # Fallback for unexpected write issue
            print(f"Warning: Failed to persist DocStore to disk: {e}")

    def set(self, doc_id: str, data: dict[str, Any]) -> None:
        self._store[doc_id] = data
        self._save()

    def set_batch(self, items: dict[str, dict[str, Any]]) -> None:
        self._store.update(items)
        self._save()

    def get(self, doc_id: str) -> dict[str, Any] | None:
        return self._store.get(doc_id)

    def get_many(self, doc_ids: list[str]) -> list[dict[str, Any]]:
        results = []
        for doc_id in doc_ids:
            item = self._store.get(doc_id)
            if item is not None:
                results.append(item)
        return results

    def delete(self, doc_id: str) -> bool:
        if doc_id in self._store:
            del self._store[doc_id]
            self._save()
            return True
        return False

    def clear(self) -> None:
        self._store.clear()
        self._save()

    def __len__(self) -> int:
        return len(self._store)
