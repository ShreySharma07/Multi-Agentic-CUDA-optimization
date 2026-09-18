"use client";

import { Canvas, useFrame } from "@react-three/fiber";
import { Html, Line, PerspectiveCamera } from "@react-three/drei";
import * as THREE from "three";
import { useMemo, useRef, type RefObject } from "react";

export type SceneMode = "hero" | "story";

type Props = {
  mode: SceneMode;
  /** 0..1 scroll progress for story mode. Read every frame, never re-renders. */
  progressRef?: RefObject<number>;
  className?: string;
};

/* ---------- geometry constants (RTX A4000: 48 SMs → 8 × 6 grid) ---------- */
const COLS = 8;
const ROWS = 6;
const SM = 0.3;
const GAP = 0.075;
const DIE_W = COLS * SM + (COLS - 1) * GAP;
const DIE_D = ROWS * SM + (ROWS - 1) * GAP;
const BOARD_W = 6.4;
const BOARD_D = 4.2;
const BOARD_T = 0.12;
const MEM_X = 2.35;
const MEM_Z = [-1.35, -0.45, 0.45, 1.35];

const BASE = new THREE.Color("#161616");
const MINT = new THREE.Color("#8ff0c6");
const COBALT = new THREE.Color("#3f5cff");
const WHITE = new THREE.Color("#f3f3f1");

function rank(i: number) {
  const x = Math.sin(i * 12.9898 + 78.233) * 43758.5453;
  return x - Math.floor(x);
}

export const STAGES = [
  {
    key: "preflight",
    label: "Pre-flight",
    title: "Profile before anyone guesses",
    body: "The kernel is compiled once, run under Nsight Compute, and its hardware counters are cached by source hash. Occupancy, SM throughput and DRAM throughput decide whether it is compute- or memory-bound.",
    metrics: [["occupancy", "41.4%"], ["compute", "23.0%"], ["dram", "10.6%"], ["verdict", "compute-bound"]],
  },
  {
    key: "rewrite",
    label: "Rewrite",
    title: "One focused change per round",
    body: "A coder agent gets the profile, the strategy hint, every past attempt this session and the closest insights from the knowledge base. It returns a full .cu file with exactly one optimization applied.",
    metrics: [["agent", "gemini-2.5-flash"], ["runtime", "google adk"], ["constraint", "float32 only"], ["output", "raw .cu"]],
  },
  {
    key: "verify",
    label: "Compile · Validate",
    title: "Wrong answers never get timed",
    body: "nvcc builds the candidate for sm_86. The binary compares GPU output to a CPU reference at 1e-3 tolerance and must print SUCCESS. Compile errors and math errors are fed back verbatim into the next round.",
    metrics: [["compiler", "nvcc -O2 -arch=sm_86"], ["tolerance", "1e-3 relative"], ["timeout", "30 s"], ["on failure", "error → next prompt"]],
  },
  {
    key: "benchmark",
    label: "Benchmark",
    title: "Measured on the GPU, not estimated",
    body: "Warm-up runs, then timed runs with cudaEvent timing. The candidate only becomes the new best if it is faster than every previous validated round. The loop stops early once gains fall under one percent.",
    metrics: [["warmup", "3 runs"], ["timed", "10 runs"], ["best", "2.37× · round 4"], ["convergence", "< 1% Δ"]],
  },
  {
    key: "reflect",
    label: "Reflect",
    title: "Every round teaches the next kernel",
    body: "A reflector agent distills the attempt into STRATEGY, INSIGHT, APPLICABLE_WHEN and AVOID_IF. It is embedded into ChromaDB and retrieved by bottleneck for the next kernel that looks like this one.",
    metrics: [["store", "chromadb"], ["space", "cosine"], ["retrieved", "top 3"], ["keyed by", "bottleneck + kernel"]],
  },
] as const;

export function stageOf(p: number) {
  const s = Math.min(STAGES.length - 1, Math.floor(p * STAGES.length));
  const t = Math.min(1, Math.max(0, p * STAGES.length - s));
  return { s, t };
}

/* ---------- board + die ---------- */
function Board() {
  const box = useMemo(() => new THREE.BoxGeometry(BOARD_W, BOARD_T, BOARD_D), []);
  const die = useMemo(() => new THREE.BoxGeometry(DIE_W + 0.22, 0.06, DIE_D + 0.22), []);
  const mem = useMemo(() => new THREE.BoxGeometry(0.62, 0.08, 0.72), []);
  return (
    <group>
      <mesh geometry={box} position={[0, -BOARD_T / 2, 0]}>
        <meshStandardMaterial color="#0a0a0a" roughness={0.9} metalness={0.2} />
      </mesh>
      <lineSegments position={[0, -BOARD_T / 2, 0]}>
        <edgesGeometry args={[box]} />
        <lineBasicMaterial color="#2c2c2c" />
      </lineSegments>
      {/* die substrate */}
      <mesh geometry={die} position={[0, 0.03, 0]}>
        <meshStandardMaterial color="#0e0e0e" roughness={0.7} metalness={0.4} />
      </mesh>
      <lineSegments position={[0, 0.03, 0]}>
        <edgesGeometry args={[die]} />
        <lineBasicMaterial color="#3a3a3a" />
      </lineSegments>
      {/* memory modules */}
      {MEM_Z.map((z) =>
        [-MEM_X, MEM_X].map((x) => (
          <group key={`${x}${z}`} position={[x, 0.04, z]}>
            <mesh geometry={mem}>
              <meshStandardMaterial color="#0d0d0d" roughness={0.8} />
            </mesh>
            <lineSegments>
              <edgesGeometry args={[mem]} />
              <lineBasicMaterial color="#333" />
            </lineSegments>
          </group>
        )),
      )}
      {/* board trace grid */}
      {Array.from({ length: 7 }, (_, i) => -BOARD_D / 2 + (i * BOARD_D) / 6).map((z) => (
        <Line
          key={`h${z}`}
          points={[[-BOARD_W / 2, 0.002, z], [BOARD_W / 2, 0.002, z]]}
          color="#181818"
          lineWidth={1}
        />
      ))}
      {Array.from({ length: 9 }, (_, i) => -BOARD_W / 2 + (i * BOARD_W) / 8).map((x) => (
        <Line
          key={`v${x}`}
          points={[[x, 0.002, -BOARD_D / 2], [x, 0.002, BOARD_D / 2]]}
          color="#181818"
          lineWidth={1}
        />
      ))}
    </group>
  );
}

function DotField() {
  const geom = useMemo(() => {
    const nx = 96;
    const nz = 64;
    const w = 13;
    const d = 9;
    const pos = new Float32Array(nx * nz * 3);
    let k = 0;
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < nz; j++) {
        const x = -w / 2 + (i / (nx - 1)) * w;
        const z = -d / 2 + (j / (nz - 1)) * d;
        pos[k++] = x;
        pos[k++] = -BOARD_T - 0.02;
        pos[k++] = z;
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    return g;
  }, []);
  return (
    <points geometry={geom}>
      <pointsMaterial color="#2a2a2a" size={0.022} sizeAttenuation transparent opacity={0.9} />
    </points>
  );
}

/* ---------- the animated core ---------- */
function Core({ mode, progressRef }: { mode: SceneMode; progressRef?: RefObject<number> }) {
  const group = useRef<THREE.Group>(null);
  const smRefs = useRef<(THREE.Mesh | null)[]>([]);
  const intensities = useRef<Float32Array>(new Float32Array(COLS * ROWS));
  const laneMats = useRef<(THREE.Material | null)[]>([]);
  const ringGroup = useRef<THREE.Group>(null);
  const ringMats = useRef<(THREE.Material | null)[]>([]);
  const markers = useRef<(THREE.Mesh | null)[]>([]);
  const packet = useRef<THREE.Mesh>(null);
  const insight = useRef<THREE.Mesh>(null);
  const kbGroup = useRef<THREE.Group>(null);
  const labels = useRef<(HTMLDivElement | null)[]>([]);
  const camTarget = useMemo(() => new THREE.Vector3(0, 0.2, 0), []);
  const smGeom = useMemo(() => new THREE.BoxGeometry(SM, 0.14, SM), []);
  const tmpColor = useMemo(() => new THREE.Color(), []);

  const smPositions = useMemo(() => {
    const out: [number, number, number][] = [];
    for (let r = 0; r < ROWS; r++) {
      for (let c = 0; c < COLS; c++) {
        out.push([
          -DIE_W / 2 + SM / 2 + c * (SM + GAP),
          0.13,
          -DIE_D / 2 + SM / 2 + r * (SM + GAP),
        ]);
      }
    }
    return out;
  }, []);

  const rings = useMemo(() => {
    const mk = (rx: number, rz: number, tilt: number, spin: number) => {
      const pts: [number, number, number][] = [];
      for (let i = 0; i <= 96; i++) {
        const a = (i / 96) * Math.PI * 2;
        pts.push([Math.cos(a) * rx, 0, Math.sin(a) * rz]);
      }
      return { pts, tilt, spin, rx, rz };
    };
    return [mk(2.3, 1.2, 0.35, 0.18), mk(1.7, 1.7, -0.5, -0.12), mk(2.9, 0.9, 0.9, 0.09)];
  }, []);

  const lanes = useMemo(() => {
    const out: { from: [number, number, number]; to: [number, number, number] }[] = [];
    MEM_Z.forEach((z) => {
      out.push({ from: [-DIE_W / 2 - 0.11, 0.08, z * 0.55], to: [-MEM_X + 0.31, 0.08, z] });
      out.push({ from: [DIE_W / 2 + 0.11, 0.08, z * 0.55], to: [MEM_X - 0.31, 0.08, z] });
    });
    return out;
  }, []);

  const registerLabel = (i: number, el: HTMLDivElement | null) => {
    labels.current[i] = el;
  };

  const setLabel = (i: number, visible: boolean) => {
    const el = labels.current[i];
    if (!el) return;
    el.style.opacity = visible ? "1" : "0";
    el.style.transform = visible ? "translateY(0)" : "translateY(6px)";
  };

  useFrame((state) => {
    const time = state.clock.elapsedTime;
    const p = mode === "story" ? (progressRef?.current ?? 0) : 0;
    const { s, t } = stageOf(p);
    const hero = mode === "hero";

    /* ---- camera + rotation ---- */
    if (group.current) {
      if (hero) {
        group.current.rotation.y = time * 0.07;
      } else {
        group.current.rotation.y = -0.75 + p * 1.5 + Math.sin(time * 0.25) * 0.03;
      }
    }
    const cam = state.camera;
    if (hero) {
      cam.position.set(0, 6.4, 9.6);
    } else {
      const lift = s === 4 ? t : 0;
      cam.position.set(0.5 - lift * 0.8, 5.3 + lift * 0.6 + Math.sin(time * 0.3) * 0.05, 8.6);
    }
    cam.lookAt(camTarget);

    /* ---- SM occupancy target ---- */
    let occ = 0.41;
    let waveCol = -1;
    let flash = 0;
    if (hero) {
      occ = 0.55 + Math.sin(time * 0.5) * 0.15;
    } else if (s === 0) {
      occ = t < 0.35 ? 0 : 0.41;
    } else if (s === 1) {
      occ = 0.12;
    } else if (s === 2) {
      waveCol = t * 1.15 * COLS;
      flash = t > 0.88 ? (t - 0.88) / 0.12 : 0;
    } else if (s === 3) {
      occ = 0.86;
    } else {
      occ = 0.6;
    }

    for (let i = 0; i < COLS * ROWS; i++) {
      const c = i % COLS;
      let target = 0;
      if (waveCol >= 0) {
        target = c < waveCol ? 0.55 : 0;
        target = Math.max(target, flash);
      } else {
        target = rank(i) < occ ? 0.55 + 0.45 * (0.5 + 0.5 * Math.sin(time * 2.2 + i * 0.7)) : 0.03;
      }
      if (hero) target *= 0.55;
      const cur = intensities.current[i];
      const nxt = cur + (target - cur) * 0.09;
      intensities.current[i] = nxt;
      const mesh = smRefs.current[i];
      if (mesh) {
        const mat = mesh.material as THREE.MeshBasicMaterial;
        const validated = s === 2 && flash > 0.4;
        tmpColor.copy(BASE).lerp(validated ? WHITE : MINT, Math.min(1, nxt));
        mat.color.copy(tmpColor);
        mesh.scale.y = 1 + nxt * 0.9;
        mesh.position.y = 0.13 + (nxt * 0.9 * 0.14) / 2;
      }
    }

    /* ---- DRAM lanes ---- */
    const laneOn = hero ? 0.35 + 0.25 * Math.sin(time * 0.9) : s === 0 && t > 0.35 ? 0.9 : s === 3 ? 0.6 : 0.12;
    laneMats.current.forEach((m, i) => {
      if (!m) return;
      const pulse = 0.5 + 0.5 * Math.sin(time * 3 + i * 0.9);
      const target = laneOn * (0.4 + 0.6 * pulse);
      m.opacity += (target - m.opacity) * 0.1;
    });

    /* ---- candidate rings (rewrite stage) ---- */
    const ringOn = hero ? 0.35 : s === 1 ? 1 : 0;
    if (ringGroup.current) {
      ringGroup.current.children.forEach((child, i) => {
        const r = rings[i];
        if (!r) return;
        child.rotation.y = time * r.spin;
        child.rotation.x = r.tilt;
      });
      ringGroup.current.position.y = 0.9 + Math.sin(time * 0.6) * 0.05;
    }
    ringMats.current.forEach((m) => {
      if (m) m.opacity += (ringOn * 0.9 - m.opacity) * 0.08;
    });
    markers.current.forEach((mk, i) => {
      if (!mk) return;
      const r = rings[i % rings.length];
      const a = time * (0.5 + i * 0.17) + i * 2.1;
      mk.position.set(Math.cos(a) * r.rx, 0, Math.sin(a) * r.rz);
      const sc = ringOn;
      mk.scale.setScalar(THREE.MathUtils.lerp(mk.scale.x, sc, 0.08));
    });

    /* ---- kernel packet (pre-flight stage) ---- */
    if (packet.current) {
      const on = !hero && s === 0;
      const y = on ? THREE.MathUtils.lerp(3.2, 0.34, Math.min(1, t / 0.35)) : 3.2;
      packet.current.position.set(0, y, 0);
      const vis = on && t < 0.98 ? 1 : 0;
      packet.current.scale.setScalar(THREE.MathUtils.lerp(packet.current.scale.x, vis, 0.12));
      packet.current.rotation.y = time * 1.4;
      packet.current.rotation.x = time * 0.9;
    }

    /* ---- insight → knowledge base (reflect stage) ---- */
    if (insight.current) {
      const on = !hero && s === 4;
      const k = on ? Math.min(1, t / 0.75) : 0;
      const ease = k * k * (3 - 2 * k);
      insight.current.position.set(
        THREE.MathUtils.lerp(0, -3.0, ease),
        THREE.MathUtils.lerp(0.35, 0.7 + 0.16 * 4, ease) + Math.sin(ease * Math.PI) * 0.9,
        THREE.MathUtils.lerp(0, 1.2, ease),
      );
      insight.current.rotation.y = time * 0.8;
      const vis = on ? 1 : 0;
      insight.current.scale.setScalar(THREE.MathUtils.lerp(insight.current.scale.x, vis, 0.1));
    }
    if (kbGroup.current) {
      const on = !hero && s === 4;
      kbGroup.current.children.forEach((c, i) => {
        const target = on ? 1 : 0;
        c.scale.x = THREE.MathUtils.lerp(c.scale.x, target, 0.08 + i * 0.01);
        c.scale.z = c.scale.x;
        c.scale.y = c.scale.x;
      });
    }

    /* ---- labels ---- */
    if (!hero) {
      setLabel(0, s === 0 && t > 0.45);
      setLabel(1, s === 0 && t > 0.6);
      setLabel(2, s === 1);
      setLabel(3, s === 1);
      setLabel(4, s === 1);
      setLabel(5, s === 2 && flash > 0.3);
      setLabel(6, s === 3);
      setLabel(7, s === 4 && t > 0.5);
    } else {
      labels.current.forEach((_, i) => setLabel(i, false));
    }
  });

  return (
    <group ref={group}>
      <Board />
      <DotField />

      {/* SMs */}
      {smPositions.map((pos, i) => (
        <mesh
          key={i}
          geometry={smGeom}
          position={pos}
          ref={(el) => {
            smRefs.current[i] = el;
          }}
        >
          <meshBasicMaterial color={BASE} toneMapped={false} />
          <lineSegments>
            <edgesGeometry args={[smGeom]} />
            <lineBasicMaterial color="#3d3d3d" />
          </lineSegments>
        </mesh>
      ))}

      {/* DRAM lanes */}
      {lanes.map((l, i) => (
        <Line
          key={i}
          points={[l.from, l.to]}
          color="#8ff0c6"
          lineWidth={1.2}
          transparent
          opacity={0.1}
          ref={(el) => {
            // drei Line ref is a Line2; grab its material
            laneMats.current[i] = (el as unknown as { material?: THREE.Material } | null)?.material ?? null;
          }}
        />
      ))}

      {/* candidate rings */}
      <group ref={ringGroup}>
        {rings.map((r, i) => (
          <group key={i}>
            <Line
              points={r.pts}
              color={i === 1 ? "#f3f3f1" : "#9a9a9a"}
              lineWidth={1}
              transparent
              opacity={0}
              ref={(el) => {
                ringMats.current[i] = (el as unknown as { material?: THREE.Material } | null)?.material ?? null;
              }}
            />
          </group>
        ))}
      </group>
      {/* markers ride on rings (world-space approx, tilt ignored for legibility) */}
      <group position={[0, 0.9, 0]}>
        {[0, 1, 2, 3, 4].map((i) => (
          <mesh
            key={i}
            ref={(el) => {
              markers.current[i] = el;
            }}
            scale={0}
          >
            <boxGeometry args={[0.11, 0.11, 0.11]} />
            <meshBasicMaterial color={i % 2 ? COBALT : MINT} toneMapped={false} />
          </mesh>
        ))}
      </group>

      {/* kernel packet */}
      <mesh ref={packet} scale={0}>
        <boxGeometry args={[0.32, 0.32, 0.32]} />
        <meshBasicMaterial color={MINT} wireframe toneMapped={false} />
      </mesh>

      {/* insight card */}
      <mesh ref={insight} scale={0}>
        <boxGeometry args={[0.7, 0.04, 0.5]} />
        <meshBasicMaterial color={WHITE} toneMapped={false} />
      </mesh>

      {/* knowledge base stack */}
      <group ref={kbGroup} position={[-3.0, 0.7, 1.2]}>
        {[0, 1, 2, 3].map((i) => (
          <group key={i} position={[0, i * 0.16, 0]} scale={0}>
            <mesh>
              <boxGeometry args={[0.7, 0.04, 0.5]} />
              <meshBasicMaterial color="#1a1a1a" />
            </mesh>
            <lineSegments>
              <edgesGeometry args={[new THREE.BoxGeometry(0.7, 0.04, 0.5)]} />
              <lineBasicMaterial color="#7a7a7a" />
            </lineSegments>
          </group>
        ))}
      </group>

      {/* anchored labels */}
      <Label index={0} register={registerLabel} position={[-DIE_W / 2 - 0.2, 0.95, -DIE_D / 2]} text="occupancy 41.4%" />
      <Label index={1} register={registerLabel} position={[MEM_X, 0.7, -1.35]} text="dram 10.6% · compute-bound" />
      <Label index={2} register={registerLabel} position={[-2.4, 1.7, 0.6]} text="float4 loads · 1.19×" />
      <Label index={3} register={registerLabel} position={[2.2, 2.1, -0.4]} text="shared-mem tiling · 2.41×" />
      <Label index={4} register={registerLabel} position={[0.4, 2.6, 1.2]} text="warp reduce · 2.37×" />
      <Label index={5} register={registerLabel} position={[0, 1.15, 0]} text="✓ success · |gpu−cpu| ≤ 1e-3" accent />
      <Label index={6} register={registerLabel} position={[0, 1.2, DIE_D / 2 + 0.3]} text="0.94 ms → 0.40 ms · 2.37×" accent />
      <Label index={7} register={registerLabel} position={[-3.0, 1.9, 1.2]} text="insight stored · chromadb" />
    </group>
  );
}

function Label({
  index,
  register,
  position,
  text,
  accent,
}: {
  index: number;
  register: (index: number, el: HTMLDivElement | null) => void;
  position: [number, number, number];
  text: string;
  accent?: boolean;
}) {
  return (
    <Html position={position} center zIndexRange={[5, 0]} style={{ pointerEvents: "none" }}>
      <div
        ref={(el) => register(index, el)}
        className="tag-box"
        style={{
          opacity: 0,
          whiteSpace: "nowrap",
          color: accent ? "var(--mint)" : undefined,
          borderColor: accent ? "rgba(143,240,198,0.4)" : undefined,
          transition: "opacity 0.5s var(--ease), transform 0.5s var(--ease)",
        }}
      >
        {text}
      </div>
    </Html>
  );
}

export default function GpuScene({ mode, progressRef, className }: Props) {
  return (
    <div className={className} style={{ position: "relative", width: "100%", height: "100%" }}>
      <Canvas
        dpr={[1, 1.75]}
        gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
        style={{ background: "transparent" }}
      >
        <PerspectiveCamera makeDefault fov={28} position={[0, 6, 9]} near={0.1} far={60} />
        <ambientLight intensity={0.6} />
        <pointLight position={[3, 5, 3]} intensity={18} color="#ffffff" />
        <pointLight position={[-4, 2, -3]} intensity={6} color="#8ff0c6" />
        <Core mode={mode} progressRef={progressRef} />
      </Canvas>
    </div>
  );
}
