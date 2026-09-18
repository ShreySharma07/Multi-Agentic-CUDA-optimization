"use client";

import Link from "next/link";
import { useState } from "react";
import { Logo } from "./Logo";
import styles from "./Nav.module.css";

export type NavUser = { name?: string | null; image?: string | null } | null;

const LINKS = [
  ["Loop", "/#loop"],
  ["How it works", "/#how"],
  ["Architecture", "/#architecture"],
  ["Results", "/#results"],
  ["Desktop", "/#desktop"],
] as const;

export function Nav({ user }: { user: NavUser }) {
  const [open, setOpen] = useState(false);
  return (
    <header className={styles.nav}>
      <div className={`wrap ${styles.inner}`}>
        <Link href="/" aria-label="KARMA home" onClick={() => setOpen(false)}>
          <Logo />
        </Link>
        <ul className={styles.links}>
          {LINKS.map(([label, href]) => (
            <li key={href}>
              <Link href={href}>{label}</Link>
            </li>
          ))}
        </ul>
        <div className={styles.right}>
          {user ? (
            <Link href="/dashboard" className={`${styles.user} ${styles.hideMobile}`}>
              {user.name?.split(" ")[0] ?? "Account"}
              {user.image ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={user.image} alt="" className={styles.avatar} />
              ) : (
                <span className={styles.avatar} />
              )}
            </Link>
          ) : (
            <Link href="/login" className={`btn btn-ghost btn-sm ${styles.hideMobile}`}>
              Log in
            </Link>
          )}
          <Link href="/early-access" className={`btn btn-white btn-sm ${styles.hideMobile}`}>
            Get early access
          </Link>
          <button
            className={styles.burger}
            aria-label="Menu"
            aria-expanded={open}
            onClick={() => setOpen((v) => !v)}
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor">
              {open ? (
                <path d="M3 3l10 10M13 3L3 13" />
              ) : (
                <path d="M2 4.5h12M2 8h12M2 11.5h12" />
              )}
            </svg>
          </button>
        </div>
      </div>
      <nav className={`${styles.sheet} ${open ? styles.sheetOpen : ""}`}>
        {LINKS.map(([label, href]) => (
          <Link key={href} href={href} onClick={() => setOpen(false)}>
            {label}
          </Link>
        ))}
        <div className={styles.sheetActions}>
          <Link href={user ? "/dashboard" : "/login"} className="btn btn-ghost btn-sm">
            {user ? "Dashboard" : "Log in"}
          </Link>
          <Link href="/early-access" className="btn btn-white btn-sm">
            Get early access
          </Link>
        </div>
      </nav>
    </header>
  );
}
