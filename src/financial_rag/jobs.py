import json
from copy import deepcopy
from datetime import UTC, datetime
from threading import RLock
from uuid import uuid4

from .storage import atomic_json


class JobStore:
    """Persist outcomes; supports SQLite backend or legacy atomic JSON files."""

    def __init__(self, path=None, *, store=None):
        self.path = path
        self.store = store
        self.lock = RLock()
        if self.store is not None:
            self.store.interrupt_stale_jobs()
            self.records = {}
            return
        self.records = json.loads(path.read_text(encoding="utf-8")) if path and path.exists() else {}
        changed = False
        for record in self.records.values():
            if record["status"] in {"queued", "running"}:
                record.update(
                    status="interrupted",
                    error="Server khởi động lại; hãy nạp lại tài liệu. Trang OCR đã lưu đệm được tái sử dụng.",
                )
                changed = True
        if changed and path:
            atomic_json(self.path, self.records)

    def create(self, name):
        if self.store is not None:
            key = uuid4().hex
            self.store.create_job(key, name)
            return key
        with self.lock:
            if sum(r["status"] in {"queued", "running"} for r in self.records.values()) >= 8:
                raise ValueError("Hàng đợi đã đủ 8 tài liệu; chờ xử lý xong rồi thử lại.")
            key = uuid4().hex
            now = datetime.now(UTC).isoformat()
            self.records[key] = {
                "id": key,
                "name": name,
                "status": "queued",
                "created_at": now,
                "updated_at": now,
            }
            # Retain the last 200 results; never evict an active job.
            finished = [
                k for k, r in self.records.items() if r["status"] not in {"queued", "running"}
            ]
            for old in finished[: max(0, len(self.records) - 200)]:
                self.records.pop(old)
            atomic_json(self.path, self.records)
            return key

    def update(self, key, **values):
        if self.store is not None:
            self.store.update_job(key, **values)
            return
        with self.lock:
            self.records[key].update(values, updated_at=datetime.now(UTC).isoformat())
            atomic_json(self.path, self.records)

    def get(self, key):
        if self.store is not None:
            return self.store.get_job(key)
        with self.lock:
            return deepcopy(self.records.get(key))

    def recent(self):
        if self.store is not None:
            return self.store.list_jobs(30)
        with self.lock:
            return deepcopy(
                sorted(self.records.values(), key=lambda r: r["created_at"], reverse=True)[:30]
            )
