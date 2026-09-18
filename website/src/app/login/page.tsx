import type { Metadata } from "next";
import Link from "next/link";
import { redirect } from "next/navigation";
import { auth, configuredProviders, signIn } from "@/auth";
import { SiteNav } from "@/components/SiteNav";
import { AsciiField } from "@/components/AsciiField";
import styles from "../auth.module.css";

export const metadata: Metadata = { title: "Log in — KARMA" };

const PROVIDERS: Record<string, { label: string; icon: React.ReactNode }> = {
  github: {
    label: "Continue with GitHub",
    icon: (
      <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
        <path d="M12 .5a12 12 0 0 0-3.8 23.4c.6.1.8-.3.8-.6v-2.2c-3.3.7-4-1.4-4-1.4-.6-1.4-1.4-1.8-1.4-1.8-1.1-.7.1-.7.1-.7 1.2.1 1.8 1.2 1.8 1.2 1.1 1.8 2.8 1.3 3.5 1 .1-.8.4-1.3.8-1.6-2.7-.3-5.5-1.3-5.5-5.9 0-1.3.5-2.4 1.2-3.2-.1-.3-.5-1.5.1-3.2 0 0 1-.3 3.3 1.2a11.5 11.5 0 0 1 6 0c2.3-1.5 3.3-1.2 3.3-1.2.7 1.7.3 2.9.1 3.2.8.8 1.2 1.9 1.2 3.2 0 4.6-2.8 5.6-5.5 5.9.4.4.8 1.1.8 2.2v3.3c0 .3.2.7.8.6A12 12 0 0 0 12 .5Z" />
      </svg>
    ),
  },
  google: {
    label: "Continue with Google",
    icon: (
      <svg viewBox="0 0 24 24" aria-hidden="true">
        <path fill="#EA4335" d="M12 10.2v3.9h5.4c-.2 1.3-1.6 3.8-5.4 3.8-3.3 0-5.9-2.7-5.9-6s2.6-6 5.9-6c1.9 0 3.1.8 3.8 1.5l2.6-2.5C16.8 3.3 14.6 2.4 12 2.4 6.7 2.4 2.4 6.7 2.4 12S6.7 21.6 12 21.6c5.5 0 9.2-3.9 9.2-9.4 0-.6-.1-1.1-.2-1.6H12Z" />
      </svg>
    ),
  },
};

export default async function LoginPage(props: PageProps<"/login">) {
  const session = await auth();
  if (session?.user) redirect("/dashboard");
  const sp = await props.searchParams;
  const next = typeof sp.next === "string" ? sp.next : "/dashboard";
  const error = typeof sp.error === "string" ? sp.error : null;

  return (
    <>
      <SiteNav />
      <main className={styles.page}>
        <div className={styles.left}>
          <AsciiField />
          <div className={styles.quote}>
            <span className="eyebrow">account</span>
            <h2 className="display-m">One login for the web console and the desktop early-access queue.</h2>
          </div>
        </div>
        <div className={styles.right}>
          <div className={styles.card}>
            <div>
              <span className="eyebrow dot">log in</span>
              <h1 className="display-m" style={{ marginTop: 14 }}>Welcome back</h1>
              <p className="body" style={{ marginTop: 10 }}>
                Sign in with an OAuth provider. We store your name, email and avatar and nothing else.
              </p>
            </div>

            {error && (
              <p className={styles.error}>
                Sign-in failed ({error}). Try again or use a different provider.
              </p>
            )}

            {configuredProviders.length === 0 ? (
              <div className={styles.warn}>
                No OAuth provider is configured yet. Add GitHub or Google credentials to{" "}
                <code>website/.env</code> (see <code>.env.example</code>) and restart the server.
                The early-access form still works without an account.
              </div>
            ) : (
              <div className={styles.providers}>
                {configuredProviders.map((id) => {
                  const p = PROVIDERS[id];
                  if (!p) return null;
                  return (
                    <form
                      key={id}
                      action={async () => {
                        "use server";
                        await signIn(id, { redirectTo: next });
                      }}
                    >
                      <button type="submit" className={`btn btn-ghost ${styles.provider}`}>
                        {p.icon}
                        <span>{p.label}</span>
                        <svg className="chevron" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.6">
                          <path d="M6 3l5 5-5 5" />
                        </svg>
                      </button>
                    </form>
                  );
                })}
              </div>
            )}

            <p className={styles.hint}>
              No account is needed to read the site. Want the desktop app?{" "}
              <Link href="/early-access" style={{ color: "var(--ink)" }}>
                Join early access →
              </Link>
            </p>
          </div>
        </div>
      </main>
    </>
  );
}
