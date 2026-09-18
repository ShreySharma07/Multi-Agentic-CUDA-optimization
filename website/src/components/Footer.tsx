import Link from "next/link";
import { Logo } from "./Logo";
import styles from "./Footer.module.css";

export function Footer() {
  return (
    <footer className={styles.footer}>
      <div className={`wrap ${styles.grid}`}>
        <div className={styles.col}>
          <h4>Product</h4>
          <ul>
            <li><Link href="/#loop">The loop</Link></li>
            <li><Link href="/#how">How it works</Link></li>
            <li><Link href="/#architecture">Architecture</Link></li>
            <li><Link href="/#desktop">Desktop</Link></li>
          </ul>
        </div>
        <div className={styles.col}>
          <h4>Research</h4>
          <ul>
            <li><Link href="/#results">Measured results</Link></li>
            <li><a href="https://github.com/ScalingIntelligence/KernelBench" target="_blank" rel="noreferrer">KernelBench</a></li>
            <li><a href="https://github.com/sgl-project/sglang" target="_blank" rel="noreferrer">SGLang kernels</a></li>
          </ul>
        </div>
        <div className={styles.col}>
          <h4>Account</h4>
          <ul>
            <li><Link href="/login">Log in</Link></li>
            <li><Link href="/early-access">Early access</Link></li>
            <li><Link href="/dashboard">Dashboard</Link></li>
          </ul>
        </div>
        <div className={styles.col}>
          <h4>Legal</h4>
          <ul>
            <li><a href="#">Privacy</a></li>
            <li><a href="#">Terms</a></li>
          </ul>
        </div>
        <div className={styles.status}>
          <div className={styles.statusRow}><span>Reference hardware</span><b>NVIDIA RTX A4000</b></div>
          <div className={styles.statusRow}><span>Architecture</span><b>Ampere · sm_86 · 48 SMs</b></div>
          <div className={styles.statusRow}><span>Toolchain</span><b>nvcc -O2 · Nsight Compute</b></div>
          <div className={styles.statusRow}><span>Desktop</span><b style={{ color: "var(--mint)" }}>early access</b></div>
        </div>
      </div>
      <div className={`wrap ${styles.bottom}`}>
        <Logo size={16} />
        <span>© {new Date().getFullYear()} KARMA · Multi-agent CUDA optimization</span>
      </div>
    </footer>
  );
}
