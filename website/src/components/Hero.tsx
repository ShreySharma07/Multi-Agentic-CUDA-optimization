"use client";

import Link from "next/link";
import dynamic from "next/dynamic";
import styles from "./Hero.module.css";
import { PEAK, SHIPPED } from "@/lib/results";

const GpuScene = dynamic(() => import("./GpuScene"), { ssr: false });

export function Hero() {
  return (
    <section className={styles.hero}>
      <div className={styles.scene} aria-hidden="true">
        <GpuScene mode="hero" />
      </div>
      <div className={styles.copy}>
        <span className="eyebrow dot rise">closed-loop cuda optimization</span>
        <h1 className="display-xl rise rise-1">
          Kernels that optimize
          <br />
          themselves on real hardware.
        </h1>
        <p className="lede rise rise-2">
          KARMA is a multi-agent loop that profiles a CUDA kernel, rewrites it, compiles it,
          validates the math against a CPU reference and benchmarks it on the GPU. Only measured,
          correct speedups survive.
        </p>
        <div className={`${styles.actions} rise rise-3`}>
          <Link href="/early-access" className="btn btn-white">
            Get early access
          </Link>
          <Link href="/#how" className="btn btn-green">
            See how it works
            <svg className="chevron" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6">
              <path d="M6 3l5 5-5 5" />
            </svg>
          </Link>
        </div>
      </div>
      <div className={styles.strip}>
        <div className={`wrap ${styles.stripInner}`}>
          <div className={styles.stat}>
            <span>peak measured speedup</span>
            <b>{PEAK.toFixed(2)}×</b>
          </div>
          <div className={styles.stat}>
            <span>kernels shipped faster</span>
            <b>{SHIPPED.length} of 8 attempted</b>
          </div>
          <div className={styles.stat}>
            <span>validation before timing</span>
            <b>100% of candidates</b>
          </div>
          <div className={styles.stat}>
            <span>reference gpu</span>
            <b>RTX A4000 · sm_86</b>
          </div>
        </div>
      </div>
    </section>
  );
}
