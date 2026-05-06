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
        record = {
            kernel_name, bottleneck, strategy_used,
            speedup, result, error, insight, round
        }
        """
        uid = hashlib.sha256(
            f"{record['kernel_name']}_{record['round']}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]

        # document is what gets embedded for semantic search
        document = (
            f"kernel={record['kernel_name']} "
            f"bottleneck={record['bottleneck']} "
            f"strategy={record.get('strategy_used','unknown')} "
            f"result={record['result']} "
            f"insight={record.get('insight','none')}"
        )

        self.collection.add(
            documents=[document],
            metadatas=[record],
            ids=[uid]
        )

    def retrieve(self, bottleneck: str, kernel_name: str, n: int = 3) -> list[dict]:
        """Semantic search for similar past optimizations."""
        if self.collection.count() == 0:
            return []

        query = f"bottleneck={bottleneck} kernel={kernel_name}"
        results = self.collection.query(
            query_texts=[query],
            n_results=min(n, self.collection.count())
        )

        if not results["metadatas"]:
            return []

        return results["metadatas"][0]  # list of dicts

    def count(self) -> int:
        return self.collection.count()