import Link from "next/link";
import styles from "./Desktop.module.css";

export function Desktop() {
  return (
    <section id="desktop" className={`section ${styles.section}`}>
      <span className="section-tag">desktop · early access</span>
      <div className="wrap">
        <div className={styles.head}>
          <h2 className="display-l">KARMA Desktop runs the loop on your own GPU</h2>
          <div className={styles.headRight}>
            <p className="lede">
              A native app around the same pipeline: point it at a .cu file or a PyTorch extension,
              watch pre-flight metrics arrive, and follow every round as it compiles, validates and
              benchmarks on the card in your machine. The knowledge base stays local.
            </p>
            <div className={styles.actions}>
              <Link href="/early-access" className="btn btn-white">
                Request early access
              </Link>
              <span className="eyebrow dot">not yet released · rolling invites</span>
            </div>
          </div>
        </div>
      </div>

      <div className={styles.stage}>
        <div className={styles.dots} aria-hidden="true" />
        <div className={styles.window} aria-label="KARMA Desktop preview">
          <div className={styles.titlebar}>
            <i /><i /><i />
            <span>karma — kernel optimizer</span>
            <span className={styles.gpu}>NVIDIA RTX A4000 · 16 GB · sm_86</span>
          </div>
          <div className={styles.app}>
            <aside className={styles.side}>
              <div>
                <div className={styles.label} style={{ marginBottom: 8 }}>quick commands</div>
                <div className={styles.cmd}>› list kernels</div>
                <div className={`${styles.cmd} ${styles.cmdOn}`}>› analyze 100_HingeLoss.cu</div>
                <div className={styles.cmd}>› optimize</div>
                <div className={styles.cmd}>› show knowledge base</div>
              </div>
              <div>
                <div className={styles.label} style={{ marginBottom: 8 }}>pre-flight</div>
                <div className={styles.metric}><span>occupancy</span><b>41.4%</b></div>
                <div className={styles.metric}><span>compute</span><b>23.0%</b></div>
                <div className={styles.metric}><span>dram</span><b>10.6%</b></div>
                <div className={styles.metric}><span>verdict</span><b>compute</b></div>
              </div>
              <div style={{ marginTop: "auto" }}>
                <div className={styles.label} style={{ marginBottom: 8 }}>model</div>
                <div className={styles.cmd} style={{ border: "1px solid var(--line-2)" }}>gemini-2.5-flash ▾</div>
              </div>
            </aside>

            <main className={styles.chat}>
              <div className={styles.msg}>
                <div className={styles.avatar}>YOU</div>
                <div className={styles.msgBody}>Analyze the hinge loss kernel, then optimize it if it looks compute-bound.</div>
              </div>
              <div className={styles.msg}>
                <div className={`${styles.avatar} ${styles.avatarAgent}`}>K</div>
                <div className={styles.msgBody}>
                  Pre-flight for <code>100_HingeLoss.cu</code>: occupancy 41.4%, compute throughput
                  23.0%, DRAM 10.6%. This kernel is compute-bound with under one full wave per SM.
                  Focus will be warp-level reduction, fast math and higher work per thread.
                  <br />
                  Starting optimization for <code>100_HingeLoss.cu</code>.
                </div>
              </div>
              <div className={styles.msg} style={{ borderBottom: 0 }}>
                <div className={`${styles.avatar} ${styles.avatarAgent}`}>K</div>
                <div className={styles.msgBody}>
                  Round 4 validated at <code>2.37×</code>. Grid-stride with four elements per thread,
                  shared-memory block reduce and a warp-synchronous tail. Saved to{" "}
                  <code>kernels/results/100_HingeLoss.cu</code>.
                </div>
              </div>
              <div className={styles.input}>
                <b>›</b> continue for one more round with float4 loads <i className={styles.cursor} />
              </div>
            </main>

            <aside className={styles.panel}>
              <div>
                <div className={styles.panelTitle}>OPTIMIZATION</div>
                <div className={styles.label} style={{ marginTop: 4 }}>100_HingeLoss.cu · round 4 / 5</div>
              </div>
              <div className={styles.bar}><i /></div>
              <div className={styles.pills}>
                <span className={`${styles.pill} ${styles.pillOk}`}>R1 · 1.42×</span>
                <span className={`${styles.pill} ${styles.pillFail}`}>R2 · compile</span>
                <span className={`${styles.pill} ${styles.pillOk}`}>R3 · 1.98×</span>
                <span className={`${styles.pill} ${styles.pillOk}`}>R4 · 2.37×</span>
                <span className={`${styles.pill} ${styles.pillActive}`}>R5 · running</span>
              </div>
              <div className={styles.stats}>
                <div className={styles.stat}><span>BEST</span><b>2.37×</b></div>
                <div className={styles.stat}><span>BASELINE</span><b>0.94 ms</b></div>
                <div className={styles.stat}><span>BEST MS</span><b>0.40 ms</b></div>
                <div className={styles.stat}><span>KB HITS</span><b>3</b></div>
              </div>
              <div className={styles.log}>
                <span>round 4: compiling...</span>
                <span>round 4: validating correctness...</span>
                <span className={styles.ok}>round 4: 0.40ms → 2.37× ✓ (validation passed)</span>
                <span className={styles.ok}>new best — saved 100_HingeLoss.cu</span>
                <span>KB: stored insight → grid-stride + warp reduce</span>
                <span>round 5: asking agent...</span>
              </div>
            </aside>
          </div>
        </div>
      </div>

      <div className={styles.features}>
        <div className={styles.feature}>
          <span className="eyebrow">local first</span>
          <h3>Your kernels never leave the machine</h3>
          <p className="body">Compilation, profiling, validation and timing run against the GPU in your workstation. Only the rewrite prompt goes to the model provider you configure.</p>
        </div>
        <div className={styles.feature}>
          <span className="eyebrow">bring a model</span>
          <h3>Swap the coder agent</h3>
          <p className="body">Gemini today through Google ADK, with the runner designed so the model behind the coder and reflector can be exchanged without touching the pipeline.</p>
        </div>
        <div className={styles.feature}>
          <span className="eyebrow">memory that compounds</span>
          <h3>A knowledge base that follows you</h3>
          <p className="body">Every round&apos;s insight is stored locally in ChromaDB and retrieved for the next kernel with the same bottleneck. The tenth kernel starts smarter than the first.</p>
        </div>
      </div>
    </section>
  );
}
