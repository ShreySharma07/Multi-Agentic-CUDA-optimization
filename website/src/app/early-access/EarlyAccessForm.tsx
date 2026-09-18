"use client";

import { useActionState } from "react";
import Link from "next/link";
import { requestEarlyAccess, type EarlyAccessState } from "./actions";
import styles from "../auth.module.css";

const GPUS = [
  "NVIDIA RTX A4000 / A5000 / A6000",
  "NVIDIA RTX 30 series (Ampere)",
  "NVIDIA RTX 40 series (Ada)",
  "NVIDIA RTX 50 series (Blackwell)",
  "NVIDIA A100",
  "NVIDIA H100 / H200",
  "NVIDIA B200 / B300",
  "Other / multiple",
];

export function EarlyAccessForm({
  defaultEmail,
  emailLocked,
}: {
  defaultEmail?: string;
  emailLocked?: boolean;
}) {
  const [state, action, pending] = useActionState<EarlyAccessState, FormData>(requestEarlyAccess, {
    status: "idle",
  });

  if (state.status === "ok") {
    return (
      <div className={styles.success}>
        <span className="eyebrow dot">request received</span>
        <b className="title-s">You are #{state.position} in the queue.</b>
        <p className="body">
          We will email {state.email} when your desktop build is ready. Invites go out in small
          batches, ordered by position.
        </p>
        <p className="body">
          <Link href="/dashboard" style={{ color: "var(--ink)" }}>
            View your status →
          </Link>
        </p>
      </div>
    );
  }

  const err = state.status === "error" ? state.fields ?? {} : {};
  // React 19 resets uncontrolled fields after an action, and <select defaultValue> only applies
  // on mount, so remount the form on every error state to re-seed it from the last submission.
  const v = state.status === "error" ? state.values : {};
  const formKey = state.status === "error" ? JSON.stringify(state.values) : "idle";

  return (
    <form key={formKey} action={action} className={styles.formGrid}>
      <div className={`field ${styles.full}`}>
        <label htmlFor="email">work email</label>
        <input
          id="email"
          name="email"
          type="email"
          required
          defaultValue={v.email ?? defaultEmail}
          readOnly={emailLocked}
          placeholder="you@lab.edu"
          autoComplete="email"
        />
        {err.email && <span className={styles.error}>{err.email}</span>}
      </div>
      <div className="field">
        <label htmlFor="gpu">primary gpu</label>
        <select id="gpu" name="gpu" required defaultValue={v.gpu ?? ""}>
          <option value="" disabled>
            Select
          </option>
          {GPUS.map((g) => (
            <option key={g} value={g}>
              {g}
            </option>
          ))}
        </select>
        {err.gpu && <span className={styles.error}>{err.gpu}</span>}
      </div>
      <div className="field">
        <label htmlFor="platform">desktop platform</label>
        <select id="platform" name="platform" required defaultValue={v.platform ?? ""}>
          <option value="" disabled>
            Select
          </option>
          <option value="linux">Linux</option>
          <option value="windows">Windows</option>
          <option value="macos">macOS (remote GPU)</option>
        </select>
        {err.platform && <span className={styles.error}>{err.platform}</span>}
      </div>
      <div className={`field ${styles.full}`}>
        <label htmlFor="useCase">what are you optimizing</label>
        <select id="useCase" name="useCase" required defaultValue={v.useCase ?? ""}>
          <option value="" disabled>
            Select
          </option>
          <option value="inference">Inference kernels (attention, MoE, GEMM)</option>
          <option value="training">Training kernels</option>
          <option value="research">Research / benchmarks (KernelBench etc.)</option>
          <option value="hpc">HPC / scientific computing</option>
          <option value="other">Something else</option>
        </select>
        {err.useCase && <span className={styles.error}>{err.useCase}</span>}
      </div>
      <div className={`field ${styles.full}`}>
        <label htmlFor="notes">anything we should know (optional)</label>
        <textarea id="notes" name="notes" defaultValue={v.notes ?? ""} placeholder="Kernel types, toolchain, constraints…" />
        {err.notes && <span className={styles.error}>{err.notes}</span>}
      </div>
      {state.status === "error" && !state.fields && <p className={`${styles.error} ${styles.full}`}>{state.message}</p>}
      <div className={styles.full} style={{ display: "flex", gap: 14, alignItems: "center", flexWrap: "wrap" }}>
        <button type="submit" className="btn btn-white" disabled={pending}>
          {pending ? "Saving…" : "Request early access"}
        </button>
        <span className={styles.hint}>Rolling invites. No spam, one email when your build is ready.</span>
      </div>
    </form>
  );
}
