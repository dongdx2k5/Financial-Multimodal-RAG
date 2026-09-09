import tempfile
from pathlib import Path
from financial_rag.retrieval.doc_store import DocStore


def test_doc_store_crud() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_path = Path(tmpdir) / "docstore.json"
        store = DocStore(storage_path=storage_path)

        # 1. Set element
        store.set("doc_1", {
            "doc_id": "doc_1",
            "modality": "image",
            "raw_content": "base64encodedimage...",
            "metadata": {"page": 1}
        })

        assert len(store) == 1
        item = store.get("doc_1")
        assert item is not None
        assert item["modality"] == "image"

        # 2. Reload from disk to verify persistence
        new_store_instance = DocStore(storage_path=storage_path)
        assert len(new_store_instance) == 1
        assert new_store_instance.get("doc_1")["raw_content"] == "base64encodedimage..."

        # 3. Batch set & get_many
        new_store_instance.set_batch({
            "doc_2": {"doc_id": "doc_2", "modality": "table", "raw_content": "<table></table>"},
            "doc_3": {"doc_id": "doc_3", "modality": "text", "raw_content": "Some text"},
        })
        items = new_store_instance.get_many(["doc_1", "doc_2", "non_existing"])
        assert len(items) == 2

        # 4. Delete
        assert new_store_instance.delete("doc_1") is True
        assert new_store_instance.get("doc_1") is None
