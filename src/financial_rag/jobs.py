import json
from copy import deepcopy
from datetime import UTC, datetime
from threading import RLock
from uuid import uuid4

from .storage import atomic_json


class JobStore:
    """Persist outcomes; interrupted uploads require explicit resubmission, not auto-replay."""

    def __init__(self, path):
        self.path, self.lock = path, RLock()
        self.records = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        changed = False
        for record in self.records.values():
            if record["status"] in {"queued", "running"}:
                record.update(
                    status="interrupted",
                    error="Server khởi động lại; hãy nạp lại tài liệu. Trang OCR đã lưu đệm được tái sử dụng.",
                )
                changed = True
        if changed:
            atomic_json(self.path, self.records)

    def create(self, name):
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
        with self.lock:
            self.records[key].update(values, updated_at=datetime.now(UTC).isoformat())
            atomic_json(self.path, self.records)

    def get(self, key):
        with self.lock:
            return deepcopy(self.records.get(key))

    def recent(self):
        with self.lock:
            return deepcopy(
                sorted(self.records.values(), key=lambda r: r["created_at"], reverse=True)[:30]
            )
