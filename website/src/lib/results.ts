/**
 * Measured results pulled from results/experiments.csv in the KARMA repo.
 * Each row is the best run for that kernel on an NVIDIA RTX A4000 (sm_86).
 */
export type ResultRow = {
  kernel: string;
  source: "own" | "kernelbench_l1" | "sglang";
  bottleneck: "memory-bound" | "compute-bound" | "unknown";
  rounds: number;
  bestSpeedup: number;
  bestRound: number;
  compileFailures: number;
  validationFailures: number;
  note?: string;
};

export const SOURCE_LABEL: Record<ResultRow["source"], string> = {
  own: "In-house",
  kernelbench_l1: "KernelBench L1",
  sglang: "SGLang",
};

export const RESULTS: ResultRow[] = [
  { kernel: "moe_lora_align_kernel.cu", source: "sglang", bottleneck: "unknown", rounds: 5, bestSpeedup: 12.79, bestRound: 3, compileFailures: 0, validationFailures: 0 },
  { kernel: "rasterizer_gpu.cu", source: "sglang", bottleneck: "unknown", rounds: 5, bestSpeedup: 6.16, bestRound: 5, compileFailures: 0, validationFailures: 0 },
  { kernel: "10_3D_tensor_matrix_multiplication.cu", source: "kernelbench_l1", bottleneck: "compute-bound", rounds: 5, bestSpeedup: 2.41, bestRound: 5, compileFailures: 0, validationFailures: 1 },
  { kernel: "100_HingeLoss.cu", source: "kernelbench_l1", bottleneck: "compute-bound", rounds: 5, bestSpeedup: 2.37, bestRound: 4, compileFailures: 1, validationFailures: 0 },
  { kernel: "kernel.cu", source: "own", bottleneck: "unknown", rounds: 5, bestSpeedup: 2.11, bestRound: 1, compileFailures: 0, validationFailures: 0 },
  { kernel: "moe_align_kernel.cu", source: "sglang", bottleneck: "unknown", rounds: 5, bestSpeedup: 2.1, bestRound: 1, compileFailures: 0, validationFailures: 0 },
  { kernel: "12_Matmul_with_diagonal_matrices_.cu", source: "kernelbench_l1", bottleneck: "memory-bound", rounds: 5, bestSpeedup: 1.19, bestRound: 4, compileFailures: 0, validationFailures: 0 },
  { kernel: "13_Matmul_for_symmetric_matrices.cu", source: "kernelbench_l1", bottleneck: "compute-bound", rounds: 5, bestSpeedup: 0, bestRound: 0, compileFailures: 0, validationFailures: 5, note: "All five rewrites failed CPU validation. Nothing shipped." },
];

export const SHIPPED = RESULTS.filter((r) => r.bestSpeedup > 0);
export const PEAK = Math.max(...SHIPPED.map((r) => r.bestSpeedup));
export const MEDIAN = (() => {
  const s = [...SHIPPED.map((r) => r.bestSpeedup)].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
})();
