import type { Metadata } from "next";
import Link from "next/link";
import { auth } from "@/auth";
import { prisma } from "@/lib/prisma";
import { SiteNav } from "@/components/SiteNav";
import { AsciiField } from "@/components/AsciiField";
import { EarlyAccessForm } from "./EarlyAccessForm";
import styles from "../auth.module.css";

export const metadata: Metadata = { title: "Early access — KARMA Desktop" };

export default async function EarlyAccessPage() {
  const session = await auth();
  const email = session?.user?.email ?? undefined;
  const existing = session?.user?.id
    ? await prisma.earlyAccess.findFirst({
        where: { OR: [{ userId: session.user.id }, ...(email ? [{ email }] : [])] },
      })
    : null;
  const position = existing
    ? await prisma.earlyAccess.count({ where: { createdAt: { lte: existing.createdAt } } })
    : null;

  return (
    <>
      <SiteNav />
      <main className={styles.page}>
        <div className={styles.left}>
          <AsciiField />
          <div className={styles.quote}>
            <span className="eyebrow">karma desktop · not yet released</span>
            <h2 className="display-m">Run the optimization loop against the GPU on your desk.</h2>
            <p className="body" style={{ maxWidth: "46ch" }}>
              Local compile, profile, validate and benchmark. A knowledge base that stays on your
              machine. Early builds target Linux workstations with Ampere or newer NVIDIA GPUs.
            </p>
          </div>
        </div>
        <div className={styles.right}>
          <div className={styles.card}>
            <div>
              <span className="eyebrow dot">early access</span>
              <h1 className="display-m" style={{ marginTop: 14 }}>Request a desktop build</h1>
              <p className="body" style={{ marginTop: 10 }}>
                {session?.user
                  ? `Signed in as ${session.user.email}. Your request will be linked to this account.`
                  : "No account required. Log in first if you want to track your position later."}
                {!session?.user && (
                  <>
                    {" "}
                    <Link href="/login?next=/early-access" style={{ color: "var(--ink)" }}>
                      Log in →
                    </Link>
                  </>
                )}
              </p>
            </div>
            {existing && position ? (
              <div className={styles.success}>
                <span className="eyebrow dot">already on the list</span>
                <b className="title-s">You are #{position} in the queue.</b>
                <p className="body">
                  Submitted {existing.createdAt.toLocaleDateString("en-US", { dateStyle: "medium" })} for{" "}
                  {existing.gpu} on {existing.platform}. Update your details below if anything changed.
                </p>
              </div>
            ) : null}
            <EarlyAccessForm defaultEmail={email} emailLocked={Boolean(email)} />
          </div>
        </div>
      </main>
    </>
  );
}
