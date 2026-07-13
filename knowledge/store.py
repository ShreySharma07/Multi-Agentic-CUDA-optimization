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
        record keys: kernel_name, bottleneck, techniques, strategy_used,
                     speedup, result, error, insight, round,
                     applicable_when, avoid_if
        """
        # Content-addressed, so re-learning the same lesson about the same kernel
        # updates in place instead of appending a near-duplicate. The old id mixed
        # in a timestamp, so every round appended forever and near-identical entries
        # crowded each other out of the top-n.
        uid = hashlib.sha256(
            f"{record.get('kernel_name','?')}"
            f"|{record.get('techniques','')}"
            f"|{record.get('result','')}"
            f"|{record.get('insight','')}".encode()
        ).hexdigest()[:16]

        # document text is what gets embedded for semantic search
        document = (
            f"kernel={record.get('kernel_name','')} "
            f"bottleneck={record.get('bottleneck','')} "
            f"techniques={record.get('techniques','')} "
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

        # upsert, not add: the id is content-addressed, so re-learning the same
        # lesson overwrites it rather than raising on a duplicate id.
        self.collection.upsert(
            documents=[document],
            metadatas=[safe_meta],
            ids=[uid]
        )

    # How much an entry's outcome is worth, independent of how similar its text is.
    # Cosine similarity alone ranked a compile_failed entry with speedup=0.0 and
    # strategy_used="unknown" into the top 3 and handed it to the coder as a
    # "relevant past optimization" -- pure noise presented as evidence.
    RESULT_WEIGHT = {
        "success": 1.0,
        "validation_failed": 0.5,   # still informative: says what breaks correctness
        "unstable": 0.5,
        "compile_failed": 0.35,     # weakly informative: says what won't build
        "dims_mismatch": 0.2,
    }

    def _score(self, meta: dict, distance: float) -> float:
        """
        Rank = semantic similarity x outcome quality x speedup value.

        A near-miss lesson from a kernel that actually went fast is worth more than
        a perfectly-similar lesson from one that failed to compile. Cosine distance
        alone cannot express that, so it is only one of three factors.
        """
        similarity = 1.0 - float(distance)          # chroma cosine distance -> similarity
        outcome = self.RESULT_WEIGHT.get(meta.get("result", ""), 0.3)

        # Speedup is measured against PyTorch (cuBLAS), so ~1.0x is excellent and
        # values above 1.0 are exceptional. Map to a modest multiplier; never zero,
        # because a fast-but-imperfect attempt still carries signal.
        try:
            sp = float(meta.get("speedup", 0.0) or 0.0)
        except (TypeError, ValueError):
            sp = 0.0
        value = 0.5 + min(sp, 1.5) / 1.5            # 0.5 (sp=0) .. 1.5 (sp>=1.5)

        return similarity * outcome * value

    def retrieve(
        self,
        bottleneck: str,
        kernel_name: str = "",
        n: int = 3,
        techniques: list[str] | None = None,
        pool: int = 12,
    ) -> list[dict]:
        """
        Retrieve past insights, ranked by relevance AND usefulness.

        Two-stage: pull a wider candidate pool by semantic similarity, then re-rank
        with _score() and keep the top n. The old version returned the raw cosine
        top-3, which surfaced failures and zero-speedup noise as "relevant past
        optimizations".
        """
        total = self.collection.count()
        if total == 0:
            return []

        # A richer query than the old "bottleneck=X kernel=Y": five tokens matched
        # against multi-sentence documents was barely doing semantic search at all.
        parts = [f"bottleneck={bottleneck}", f"kernel={kernel_name}"]
        if techniques:
            parts.append("techniques=" + ",".join(techniques))
        query = " ".join(parts)

        results = self.collection.query(
            query_texts=[query],
            n_results=min(pool, total),
            include=["metadatas", "distances"],
        )
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]
        if not metas:
            return []

        ranked = sorted(
            zip(metas, dists),
            key=lambda md: self._score(md[0], md[1]),
            reverse=True,
        )
        return [m for m, _ in ranked[:n]]

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