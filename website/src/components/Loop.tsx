"use client";

import { useEffect, useState } from "react";
import styles from "./Loop.module.css";

const ITEMS = [
  {
    title: "Find the bottleneck",
    body:
      "Before the agent writes a line, the kernel runs under Nsight Compute. Occupancy, SM throughput and DRAM throughput are cached by source hash in Redis and turned into a verdict: memory-bound, compute-bound or low-occupancy.",
  },
  {
    title: "Try many rewrites",
    body:
      "Each round the coder agent applies exactly one optimization suited to the verdict: coalesced or float4 loads and shared-memory tiling for memory-bound kernels; fast math, unrolling and register pressure for compute-bound ones. Every candidate is compiled and validated against a CPU reference.",
  },
  {
    title: "Ship the measured winner",
    body:
      "Candidates are timed with cudaEvent over warm-up and timed runs. Only a validated kernel that beats the current best is kept. The loop stops early when gains fall under one percent, and the winner is written back as a full .cu file.",
  },
];

export function Loop() {
  const [open, setOpen] = useState(0);
  const [auto, setAuto] = useState(true);

  useEffect(() => {
    if (!auto) return;
    const id = setInterval(() => setOpen((v) => (v + 1) % ITEMS.length), 6500);
    return () => clearInterval(id);
  }, [auto]);

  return (
    <section id="loop" className={`section ${styles.section}`}>
      <span className="section-tag">karma loop</span>
      <div className={`wrap ${styles.grid}`}>
        <div className={styles.left}>
          <h2 className="display-l">Why a closed loop beats a clever prompt</h2>
          <p className="lede">
            A model can guess a faster kernel. It cannot know it is faster, or that it is still
            correct, without running it. KARMA puts the compiler, the profiler and the GPU inside
            the loop so every claim is a measurement.
          </p>
          <div className={styles.acc}>
            {ITEMS.map((it, i) => (
              <div key={it.title} className={`${styles.item} ${open === i ? styles.itemOpen : ""}`}>
                <button
                  className={styles.head}
                  aria-expanded={open === i}
                  onClick={() => {
                    setOpen(i);
                    setAuto(false);
                  }}
                >
                  {it.title}
                  <svg viewBox="0 0 14 14" fill="none" stroke="currentColor" strokeWidth="1.2">
                    <path d="M7 1v12M1 7h12" />
                  </svg>
                </button>
                <div className={styles.bodyWrap}>
                  <div className={styles.bodyInner}>
                    <p className="body">{it.body}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className={styles.canvas} aria-hidden="true">
          <div className={`${styles.art} ${open === 0 ? styles.artOn : ""}`}>
            <ArtBottleneck />
          </div>
          <div className={`${styles.art} ${open === 1 ? styles.artOn : ""}`}>
            <ArtPaths />
          </div>
          <div className={`${styles.art} ${open === 2 ? styles.artOn : ""}`}>
            <ArtShip />
          </div>
          <div className={styles.progress}>
            {ITEMS.map((_, i) => (
              <i key={i} className={i === open ? styles.on : ""} />
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* Line-art illustrations. Thin white strokes, mono tags, dark boxes. */
/* ------------------------------------------------------------------ */

function Tag({ x, y, text, w }: { x: number; y: number; text: string; w?: number }) {
  const width = w ?? text.length * 7.6 + 14;
  return (
    <g>
      <rect x={x} y={y - 10} width={width} height={20} fill="#141414" stroke="#2a2a2a" />
      <text
        x={x + 7}
        y={y + 4}
        fill="#c9c9c5"
        fontFamily="var(--font-mono)"
        fontSize="11.5"
        letterSpacing="0.3"
      >
        {text}
      </text>
    </g>
  );
}

function ArtBottleneck() {
  return (
    <svg viewBox="0 0 800 520" preserveAspectRatio="xMidYMid slice">
      <defs>
        <linearGradient id="fade" x1="0" x2="1">
          <stop offset="0" stopColor="#f3f3f1" stopOpacity="0" />
          <stop offset="0.5" stopColor="#f3f3f1" stopOpacity="1" />
          <stop offset="1" stopColor="#f3f3f1" stopOpacity="0" />
        </linearGradient>
      </defs>
      {/* axis */}
      <line x1="0" y1="290" x2="800" y2="290" stroke="url(#fade)" strokeWidth="1.2" />
      <line x1="400" y1="40" x2="400" y2="480" stroke="#3a3a3a" strokeWidth="1" strokeDasharray="4 6" />
      <line x1="60" y1="235" x2="740" y2="235" stroke="#2c2c2c" strokeDasharray="4 6" />
      {/* big compute ellipse */}
      <ellipse cx="400" cy="290" rx="110" ry="210" fill="none" stroke="#f3f3f1" strokeWidth="1.4" opacity="0.9" />
      <ellipse cx="400" cy="290" rx="150" ry="210" fill="none" stroke="#5a5a5a" strokeWidth="1" />
      {/* small dram rings */}
      <ellipse cx="672" cy="290" rx="26" ry="70" fill="none" stroke="#9a9a9a" strokeWidth="1.2" />
      <ellipse cx="700" cy="290" rx="26" ry="70" fill="none" stroke="#7a7a7a" strokeWidth="1" />
      <ellipse cx="728" cy="290" rx="26" ry="70" fill="none" stroke="#5a5a5a" strokeWidth="1" />
      <ellipse cx="92" cy="290" rx="24" ry="66" fill="none" stroke="#6a6a6a" strokeWidth="1" />
      {/* markers */}
      <rect x="292" y="284" width="10" height="10" fill="#8ff0c6" />
      <rect x="586" y="284" width="10" height="10" fill="#3f5cff" />
      <g className={styles.slide}>
        <circle cx="400" cy="290" r="9" fill="#e8e8e6" />
      </g>
      <circle cx="565" cy="235" r="9" fill="#f3f3f1" className={styles.blink} />
      <Tag x={240} y={212} text="COMPUTE 23.0%" />
      <Tag x={560} y={345} text="DRAM 10.6%" />
      <Tag x={412} y={70} text="OCCUPANCY 41.4%" />
      <Tag x={414} y={470} text="VERDICT: COMPUTE-BOUND" />
    </svg>
  );
}

function ArtPaths() {
  const rays = [
    { a: -34, len: 560, v: "1.19×", c: "#8ff0c6", ok: true },
    { a: -18, len: 520, v: "0.92×", c: "#3f5cff", ok: false },
    { a: -4, len: 600, v: "2.37×", c: "#8ff0c6", ok: true },
    { a: 12, len: 500, v: "compile ✕", c: "#5a5a5a", ok: false },
    { a: 28, len: 580, v: "2.41×", c: "#8ff0c6", ok: true },
    { a: 44, len: 470, v: "validate ✕", c: "#5a5a5a", ok: false },
  ];
  const ox = 780;
  const oy = 120;
  return (
    <svg viewBox="0 0 800 520" preserveAspectRatio="xMidYMid slice">
      {/* dashed guides */}
      <line x1="120" y1="0" x2="330" y2="520" stroke="#2c2c2c" strokeDasharray="6 8" />
      <line x1="0" y1="330" x2="800" y2="40" stroke="#242424" strokeDasharray="6 8" />
      {rays.map((r, i) => {
        const rad = (r.a * Math.PI) / 180;
        const x2 = ox - Math.cos(rad) * r.len;
        const y2 = oy + Math.sin(rad) * r.len;
        const mx = ox - Math.cos(rad) * (r.len * 0.55);
        const my = oy + Math.sin(rad) * (r.len * 0.55);
        return (
          <g key={i}>
            <line x1={ox} y1={oy} x2={x2} y2={y2} stroke={r.ok ? "#d9d9d6" : "#4a4a4a"} strokeWidth={r.ok ? 1.4 : 1} />
            <ellipse cx={mx} cy={my} rx="14" ry="34" fill="none" stroke={r.ok ? "#9a9a9a" : "#3a3a3a"} transform={`rotate(${r.a} ${mx} ${my})`} />
            <rect x={x2 - 5} y={y2 - 5} width="10" height="10" fill={r.c} transform={`rotate(45 ${x2} ${y2})`} />
            <Tag x={x2 + (x2 > 400 ? -90 : 16)} y={y2 - 22} text={r.v} />
          </g>
        );
      })}
      {/* orbiting candidate ring */}
      <g className={styles.orbit}>
        <ellipse cx="420" cy="300" rx="120" ry="52" fill="none" stroke="#7a7a7a" strokeWidth="1.2" transform="rotate(-28 420 300)" />
        <rect x="292" y="292" width="9" height="9" fill="#8ff0c6" transform="rotate(-28 420 300)" />
        <rect x="532" y="292" width="9" height="9" fill="#3f5cff" transform="rotate(-28 420 300)" />
      </g>
      <g className={styles.orbitRev}>
        <ellipse cx="420" cy="300" rx="70" ry="160" fill="none" stroke="#4a4a4a" strokeWidth="1" transform="rotate(20 420 300)" />
      </g>
      <Tag x={30} y={480} text="ROUND 4 / 5 · ONE CHANGE PER ROUND" />
    </svg>
  );
}

function ArtShip() {
  const rays = Array.from({ length: 22 }, (_, i) => (i * 360) / 22);
  return (
    <svg viewBox="0 0 800 520" preserveAspectRatio="xMidYMid slice">
      {rays.map((a) => {
        const rad = (a * Math.PI) / 180;
        return (
          <line
            key={a}
            x1="400"
            y1="190"
            x2={400 + Math.cos(rad) * 900}
            y2={190 + Math.sin(rad) * 900}
            stroke="#2a2a2a"
            strokeDasharray="5 9"
          />
        );
      })}
      {[60, 110, 160, 210, 260, 310].map((r, i) => (
        <ellipse
          key={r}
          cx="400"
          cy="300"
          rx={r * 1.55}
          ry={r * 0.42}
          fill="none"
          stroke={i === 5 ? "#c8c8c5" : "#4a4a4a"}
          strokeWidth={i === 5 ? 1.4 : 1}
        />
      ))}
      <line x1="400" y1="190" x2="400" y2="470" stroke="#7a7a7a" strokeWidth="1.2" />
      <line x1="0" y1="190" x2="800" y2="190" stroke="#3a3a3a" strokeDasharray="4 6" />
      <g className={styles.pulse}>
        <circle cx="400" cy="190" r="16" fill="none" stroke="#8ff0c6" />
      </g>
      <rect x="394" y="184" width="12" height="12" fill="#f3f3f1" transform="rotate(45 400 190)" />
      <text x="428" y="196" fill="#c9c9c5" fontFamily="var(--font-mono)" fontSize="15" letterSpacing="1.5">
        ✓ READY TO SHIP
      </text>
      <Tag x={470} y={240} text="VALIDATED · SUCCESS" />
      <Tag x={220} y={380} text="0.94 MS → 0.40 MS" />
      <Tag x={470} y={380} text="2.37× · ROUND 4" />
      <Tag x={30} y={480} text="kernels/results/100_HingeLoss.cu" w={270} />
    </svg>
  );
}
