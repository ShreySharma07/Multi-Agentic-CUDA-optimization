# knowledge/store.py
import chromadb
import hashlib
import json
from pathlib import Path
from datetime import datetime


class KnowledgeBase:
    def __init__(self, persist_dir: str = "knowledge/db"):
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="karma_optimizations",
            metadata={"hnsw:space": "cosine"}
        )

    def store(self, record: dict):
        """
        Store one optimization insight.
        record keys: kernel_name, bottleneck, strategy_used,
                     speedup, result, error, insight, round,
                     applicable_when, avoid_if
        """
        uid = hashlib.sha256(
            f"{record.get('kernel_name','?')}_{record.get('round',0)}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        # document text is what gets embedded for semantic search
        document = (
            f"kernel={record.get('kernel_name','')} "
            f"bottleneck={record.get('bottleneck','')} "
            f"strategy={record.get('strategy_used','unknown')} "
            f"result={record.get('result','')} "
            f"speedup={record.get('speedup', 0)} "
            f"insight={record.get('insight','none')} "
            f"applicable_when={record.get('applicable_when','unknown')} "
            f"avoid_if={record.get('avoid_if','unknown')}"
        )

        # chromadb metadata values must be str/int/float/bool
        safe_meta = {}
        for k, v in record.items():
            if isinstance(v, (str, int, float, bool)):
                safe_meta[k] = v
            else:
                safe_meta[k] = str(v)

        self.collection.add(
            documents=[document],
            metadatas=[safe_meta],
            ids=[uid]
        )

    def retrieve(self, bottleneck: str, kernel_name: str = "", n: int = 3) -> list[dict]:
        """Semantic search for similar past optimizations."""
        if self.collection.count() == 0:
            return []

        query = f"bottleneck={bottleneck} kernel={kernel_name}"
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n, self.collection.count())
        )

        if not results or not results.get("metadatas"):
            return []

        return results["metadatas"][0]  # list of dicts

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        """Delete all entries — useful for ablation runs."""
        self.client.delete_collection("karma_optimizations")
        self.collection = self.client.get_or_create_collection(
            name="karma_optimizations",
            metadata={"hnsw:space": "cosine"}
        )


if __name__ == "__main__":
    # quick smoke test
    kb = KnowledgeBase()
    print(f"KB has {kb.count()} entries")
    kb.store({
        "kernel_name": "test_kernel",
        "bottleneck": "memory-bound",
        "strategy_used": "shared memory tiling",
        "speedup": 1.5,
        "result": "success",
        "insight": "reduces redundant global reads",
        "round": 1,
        "applicable_when": "threads access same data",
        "avoid_if": "inputSize > 48KB",
    })
    print(f"KB now has {kb.count()} entries")
    results = kb.retrieve("memory-bound", "test_kernel")
    print(f"Retrieved: {results}")