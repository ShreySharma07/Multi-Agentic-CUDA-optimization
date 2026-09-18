import type { Metadata } from "next";
import Link from "next/link";
import { redirect } from "next/navigation";
import { auth, signOut } from "@/auth";
import { prisma } from "@/lib/prisma";
import { SiteNav } from "@/components/SiteNav";
import { AsciiField } from "@/components/AsciiField";
import styles from "../auth.module.css";

export const metadata: Metadata = { title: "Dashboard — KARMA" };

export default async function DashboardPage() {
  const session = await auth();
  if (!session?.user) redirect("/login?next=/dashboard");
  const user = session.user;

  const request = await prisma.earlyAccess.findFirst({
    where: { OR: [{ userId: user.id }, ...(user.email ? [{ email: user.email }] : [])] },
  });
  const position = request
    ? await prisma.earlyAccess.count({ where: { createdAt: { lte: request.createdAt } } })
    : null;
  const total = await prisma.earlyAccess.count();

  return (
    <>
      <SiteNav />
      <main className={styles.page}>
        <div className={styles.left}>
          <AsciiField />
          <div className={styles.quote}>
            <span className="eyebrow">account</span>
            <h2 className="display-m">Your place in the desktop early-access queue.</h2>
          </div>
        </div>
        <div className={styles.right}>
          <div className={styles.card}>
            <div className={styles.identity}>
              {user.image ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={user.image} alt="" />
              ) : (
                <i />
              )}
              <div>
                <b className="title-s" style={{ fontWeight: 500 }}>{user.name ?? "Signed in"}</b>
                <small>{user.email}</small>
              </div>
            </div>

            <div className={styles.meta}>
              <div className={styles.metaCell}>
                <span>early access</span>
                <b style={{ color: request ? "var(--mint)" : "var(--ink-3)" }}>
                  {request ? "requested" : "not requested"}
                </b>
              </div>
              <div className={styles.metaCell}>
                <span>queue position</span>
                <b>{position ? `#${position} of ${total}` : "—"}</b>
              </div>
              <div className={styles.metaCell}>
                <span>gpu</span>
                <b style={{ fontSize: 15 }}>{request?.gpu ?? "—"}</b>
              </div>
              <div className={styles.metaCell}>
                <span>platform</span>
                <b>{request?.platform ?? "—"}</b>
              </div>
            </div>

            <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
              <Link href="/early-access" className="btn btn-white btn-sm">
                {request ? "Update request" : "Request early access"}
              </Link>
              <form
                action={async () => {
                  "use server";
                  await signOut({ redirectTo: "/" });
                }}
              >
                <button type="submit" className="btn btn-ghost btn-sm">
                  Log out
                </button>
              </form>
            </div>

            <p className={styles.hint}>
              Desktop builds ship in small batches ordered by queue position. The web console and
              this dashboard stay free.
            </p>
          </div>
        </div>
      </main>
    </>
  );
}
