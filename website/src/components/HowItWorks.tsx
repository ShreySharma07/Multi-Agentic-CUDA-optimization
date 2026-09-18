"use client";

import dynamic from "next/dynamic";
import { useEffect, useRef, useState } from "react";
import { STAGES, stageOf } from "./GpuScene";
import styles from "./HowItWorks.module.css";

const GpuScene = dynamic(() => import("./GpuScene"), { ssr: false });

export function HowItWorks() {
  const outer = useRef<HTMLDivElement>(null);
  const progressRef = useRef(0);
  const barRef = useRef<HTMLElement>(null);
  const [stage, setStage] = useState(0);

  useEffect(() => {
    let raf = 0;
    const update = () => {
      raf = 0;
      const el = outer.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const total = rect.height - window.innerHeight;
      const p = Math.min(1, Math.max(0, (-rect.top + 64) / Math.max(1, total)));
      progressRef.current = p;
      if (barRef.current) barRef.current.style.transform = `scaleX(${p})`;
      const s = stageOf(p).s;
      setStage((prev) => (prev === s ? prev : s));
    };
    const onScroll = () => {
      if (!raf) raf = requestAnimationFrame(update);
    };
    update();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
      if (raf) cancelAnimationFrame(raf);
    };
  }, []);

  const st = STAGES[stage];

  return (
    <section id="how" className={`section ${styles.outer}`} ref={outer}>
      <div className={styles.sticky}>
        <div className={styles.left}>
          <span className="eyebrow">how it works · one round</span>
          <h2 className={styles.title}>Six stages. One measured winner.</h2>
          <ol className={styles.steps}>
            {STAGES.map((s, i) => (
              <li key={s.key} className={`${styles.step} ${i === stage ? styles.stepActive : ""}`}>
                <span>{String(i + 1).padStart(2, "0")}</span>
                <span>{s.label}</span>
              </li>
            ))}
          </ol>
          <div className={styles.detail} key={st.key}>
            <h3 className="title-s rise">{st.title}</h3>
            <p className="body rise rise-1">{st.body}</p>
            <div className={`${styles.metrics} rise rise-2`}>
              {st.metrics.map(([k, v]) => (
                <div key={k} className={styles.metric}>
                  <span>{k}</span>
                  <b>{v}</b>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className={styles.right}>
          <span className={`tag-box ${styles.stageTag}`}>
            stage {String(stage + 1).padStart(2, "0")} / {STAGES.length} · {st.label}
          </span>
          <GpuScene mode="story" progressRef={progressRef} />
          <div className={styles.progress} aria-hidden="true">
            <i ref={barRef} />
          </div>
        </div>
      </div>
    </section>
  );
}
