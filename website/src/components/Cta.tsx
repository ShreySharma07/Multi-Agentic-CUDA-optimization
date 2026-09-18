import Link from "next/link";
import { AsciiField } from "./AsciiField";
import styles from "./Cta.module.css";

export function Cta() {
  return (
    <section className={styles.cta}>
      <div className={styles.field} aria-hidden="true">
        <AsciiField />
      </div>
      <div className={`wrap ${styles.inner}`}>
        <span className="eyebrow dot">your kernel, measured</span>
        <h2 className="display-l">Stop guessing what the GPU will do</h2>
        <p className="lede">
          Bring a CUDA kernel or a PyTorch extension. KARMA profiles it, rewrites it, and ships
          only what compiles, validates and measures faster.
        </p>
        <div className={styles.actions}>
          <Link href="/early-access" className="btn btn-white">
            Get early access
          </Link>
          <Link href="/login" className="btn btn-ghost">
            Log in
          </Link>
        </div>
      </div>
    </section>
  );
}
