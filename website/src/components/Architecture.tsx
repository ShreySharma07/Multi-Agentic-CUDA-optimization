import { AsciiField } from "./AsciiField";
import styles from "./Architecture.module.css";

const ROWS = [
  {
    title: "Pre-flight: profile before you guess",
    body: "The source is hashed and looked up in Redis. On a miss it is compiled with nvcc, run under Nsight Compute with the full metric set, and parsed for achieved occupancy, SM throughput and DRAM throughput. The result is cached for a week.",
    chips: ["nvcc", "ncu --set full", "Redis", "sha256 keyed"],
  },
  {
    title: "A coder agent with tools, not a chat",
    body: "Built on Google's Agent Development Kit, the coder can list and read kernel files. Its prompt carries the hardware profile, the bottleneck-specific strategy hint, the round history and retrieved knowledge-base insights. Output must be a raw, compilable .cu file starting at #include.",
    chips: ["Google ADK", "Gemini 2.5 Flash", "getFiles · readFile", "retry with backoff"],
  },
  {
    title: "Compile, validate, benchmark",
    body: "Every candidate goes through the same deterministic gates. Compile errors and failed validations are written back into the code as comments so the next round fixes those exact lines first. Timing is only recorded for kernels that print SUCCESS.",
    chips: ["nvcc -O2 -arch=sm_86 -lineinfo", "CPU reference · 1e-3", "cudaEvent timing", "warm-up + timed runs"],
  },
  {
    title: "Reflect into a knowledge base",
    body: "A reflector agent turns each round into a structured record: what was tried, why it worked or failed, when to apply it again, when to avoid it. Records live in a persistent ChromaDB collection and are retrieved by cosine similarity on bottleneck and kernel name.",
    chips: ["ChromaDB", "cosine", "STRATEGY · INSIGHT · AVOID_IF", "cross-kernel memory"],
  },
  {
    title: "A live surface for humans",
    body: "A FastAPI WebSocket server drives the chat and the optimization panel: pre-flight metrics, per-round pills, speedup, and the saved best kernel. Two explicit commands, analyze and optimize, bypass the agent; everything else is natural language.",
    chips: ["FastAPI", "WebSocket", "nvidia-smi detection", "KernelBench · SGLang runners"],
  },
];

const PIPE = [
  ["01", "Pre-flight", "hash → cache → ncu"],
  ["02", "Retrieve", "top-3 insights by bottleneck"],
  ["03", "Rewrite", "one optimization · raw .cu"],
  ["04", "Compile", "nvcc · errors fed back"],
  ["05", "Validate", "GPU vs CPU · SUCCESS"],
  ["06", "Benchmark · Reflect", "cudaEvent · store insight"],
];

export function Architecture() {
  return (
    <section id="architecture" className={`section ${styles.section}`}>
      <span className="section-tag">architecture</span>
      <div className="wrap">
        <div className={styles.intro}>
          <h2 className="display-l">An agent reasoning across a deterministic pipeline</h2>
          <p className="lede">
            The model only does the part models are good at: proposing a rewrite. Everything that
            decides whether the rewrite ships is a program with an exit code.
          </p>
        </div>

        <div className={styles.grid}>
          <div className={styles.art}>
            <AsciiField />
            <h3 className={`display-m ${styles.artTitle}`}>
              Profile, propose, prove.
              <br />
              Then remember.
            </h3>
          </div>
          <div className={styles.rows}>
            {ROWS.map((r) => (
              <div key={r.title} className={styles.row}>
                <h3>{r.title}</h3>
                <p className="body">{r.body}</p>
                <div className="chip-row">
                  {r.chips.map((c) => (
                    <span key={c} className="chip">
                      {c}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className={styles.pipeline}>
          {PIPE.map(([n, t, d]) => (
            <div key={n} className={styles.pipeStep}>
              <small>step {n}</small>
              <b>{t}</b>
              <span>{d}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
