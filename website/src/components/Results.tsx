"use client";

import { useState } from "react";
import { MEDIAN, PEAK, RESULTS, SHIPPED, SOURCE_LABEL } from "@/lib/results";
import styles from "./Results.module.css";

export function Results() {
  const [hover, setHover] = useState<number | null>(null);

  // --- chart geometry (horizontal bars, single series) ---
  const W = 640;
  const rowH = 40;
  const padL = 250;
  const padR = 64;
  const padT = 12;
  const rows = RESULTS;
  const H = padT + rows.length * rowH + 30;
  const max = 14;
  const x = (v: number) => padL + (v / max) * (W - padL - padR);
  const ticks = [0, 1, 2, 4, 6, 8, 10, 12, 14];
  const totalRounds = RESULTS.reduce((a, r) => a + r.rounds, 0);
  const valFails = RESULTS.reduce((a, r) => a + r.validationFailures, 0);
  const compFails = RESULTS.reduce((a, r) => a + r.compileFailures, 0);

  const h = hover !== null ? rows[hover] : null;

  return (
    <section id="results" className={`section ${styles.section}`}>
      <span className="section-tag">measured</span>
      <div className="wrap">
        <div className={styles.head}>
          <h2 className="display-l">Measured, not estimated</h2>
          <p className="lede">
            Every number below came from a binary that compiled, printed SUCCESS against a CPU
            reference, and was timed with cudaEvent on an RTX A4000. Failures are listed too,
            because a loop that cannot fail is not measuring anything.
          </p>
        </div>

        <div className={styles.tiles}>
          <div className={styles.tile}>
            <span>peak speedup</span>
            <b>{PEAK.toFixed(2)}×</b>
            <small>moe_lora_align_kernel.cu · SGLang</small>
          </div>
          <div className={styles.tile}>
            <span>median shipped speedup</span>
            <b>{MEDIAN.toFixed(2)}×</b>
            <small>across {SHIPPED.length} kernels that beat baseline</small>
          </div>
          <div className={styles.tile}>
            <span>rounds executed</span>
            <b>{totalRounds}</b>
            <small>{compFails} compile failures · {valFails} validation failures</small>
          </div>
          <div className={styles.tile}>
            <span>incorrect kernels shipped</span>
            <b>0</b>
            <small>validation gates every candidate</small>
          </div>
        </div>

        <div className={styles.body}>
          <div className={styles.chartWrap}>
            <div className={styles.chartTitle}>
              <h3>Best validated speedup per kernel</h3>
              <span className="mono" style={{ color: "var(--ink-4)" }}>
                × over baseline · higher is better
              </span>
            </div>
            <svg
              className={styles.chart}
              viewBox={`0 0 ${W} ${H}`}
              role="img"
              aria-label="Horizontal bar chart of best validated speedup per kernel"
              onMouseLeave={() => setHover(null)}
            >
              <g className={styles.grid}>
                {ticks.map((t) => (
                  <line key={t} x1={x(t)} x2={x(t)} y1={padT} y2={H - 28} />
                ))}
              </g>
              <line className={styles.baseline} x1={x(1)} x2={x(1)} y1={padT} y2={H - 28} />
              <text className={styles.axisLabel} x={x(1)} y={H - 10} textAnchor="middle">
                1× baseline
              </text>
              {ticks
                .filter((t) => t !== 1 && t !== 0)
                .map((t) => (
                  <text key={t} className={styles.axisLabel} x={x(t)} y={H - 10} textAnchor="middle">
                    {t}×
                  </text>
                ))}
              {rows.map((r, i) => {
                const y = padT + i * rowH + 8;
                const bh = rowH - 16;
                const name = r.kernel.replace(".cu", "");
                const short = name.length > 30 ? name.slice(0, 29) + "…" : name;
                const v = r.bestSpeedup;
                return (
                  <g
                    key={r.kernel}
                    className={hover === i ? styles.rowHover : ""}
                    onMouseEnter={() => setHover(i)}
                  >
                    <rect className={styles.barHit} x={0} y={padT + i * rowH} width={W} height={rowH} />
                    <text className={styles.kernelLabel} x={padL - 12} y={y + bh / 2 + 4} textAnchor="end">
                      {short}
                    </text>
                    {v > 0 ? (
                      <>
                        <rect className={styles.bar} x={x(0)} y={y} width={Math.max(0, x(v) - x(0) - 4)} height={bh} />
                        <rect className={styles.barCap} x={x(v) - 4} y={y} width={4} height={bh} rx={2} />
                      </>
                    ) : (
                      <rect className={styles.barZero} x={x(0)} y={y} width={x(1) - x(0)} height={bh} />
                    )}
                    <text className={styles.valueLabel} x={x(v > 0 ? v : 1) + 10} y={y + bh / 2 + 4}>
                      {v > 0 ? `${v.toFixed(2)}×` : "no valid kernel"}
                    </text>
                  </g>
                );
              })}
            </svg>
            {h && hover !== null && (
              <div
                className={styles.tooltip}
                style={{
                  left: `${((x(Math.max(1, h.bestSpeedup)) / W) * 100).toFixed(2)}%`,
                  top: `${((padT + hover * rowH + 8) / H) * 100 + 8}%`,
                }}
              >
                <b>{h.kernel}</b>
                <span>{SOURCE_LABEL[h.source]} · {h.bottleneck}</span>
                <span>
                  best {h.bestSpeedup > 0 ? `${h.bestSpeedup.toFixed(2)}× in round ${h.bestRound}` : "—"} · {h.rounds} rounds
                </span>
                <span>
                  {h.compileFailures} compile ✕ · {h.validationFailures} validate ✕
                </span>
              </div>
            )}
          </div>

          <div>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th>Kernel</th>
                  <th>Bottleneck</th>
                  <th>Fails</th>
                  <th style={{ textAlign: "right" }}>Best</th>
                </tr>
              </thead>
              <tbody>
                {RESULTS.map((r) => (
                  <tr key={r.kernel}>
                    <td>
                      <span className={styles.kernel}>{r.kernel}</span>
                      <span className={styles.src}>{SOURCE_LABEL[r.source]}</span>
                    </td>
                    <td className="mono-lc" style={{ color: "var(--ink-3)" }}>{r.bottleneck}</td>
                    <td className="mono-lc" style={{ color: "var(--ink-3)" }}>
                      {r.compileFailures}c · {r.validationFailures}v
                    </td>
                    <td className={`${styles.speed} ${r.bestSpeedup === 0 ? styles.zero : ""}`}>
                      {r.bestSpeedup > 0 ? `${r.bestSpeedup.toFixed(2)}×` : "0 shipped"}
                      {r.bestRound > 0 && <span className={styles.src}>round {r.bestRound}</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className={styles.note}>
              13_Matmul_for_symmetric_matrices produced five rewrites that all failed CPU
              validation. KARMA shipped nothing for it. That is the gate doing its job.
            </div>
          </div>
        </div>

        <div className={styles.foot}>
          <span>source: results/experiments.csv · best run per kernel</span>
          <span>agent: gemini-2.5-flash via google adk · 5 rounds · nvcc -O2 -arch=sm_86</span>
        </div>
      </div>
    </section>
  );
}
