import styles from "./Marquee.module.css";

const ITEMS = [
  "nvcc",
  "nsight compute",
  "cuda 12 · sm_86",
  "kernelbench level 1",
  "sglang kernels",
  "google adk",
  "gemini 2.5 flash",
  "chromadb",
  "redis",
  "fastapi · websocket",
  "pytorch reference",
  "cudaEvent timing",
];

export function Marquee() {
  const list = [...ITEMS, ...ITEMS];
  return (
    <div className={styles.wrap} aria-hidden="true">
      <div className={styles.track}>
        {list.map((t, i) => (
          <span key={i}>{t}</span>
        ))}
      </div>
    </div>
  );
}
