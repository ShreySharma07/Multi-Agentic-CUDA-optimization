import NextAuth from "next-auth";
import type { Provider } from "next-auth/providers";
import GitHub from "next-auth/providers/github";
import Google from "next-auth/providers/google";
import { PrismaAdapter } from "@auth/prisma-adapter";
import { prisma } from "@/lib/prisma";

const providers: Provider[] = [];

/** A variable counts as configured unless empty or the literal placeholder "skip". */
const configured = (v: string | undefined) => Boolean(v && v.trim() && v.trim() !== "skip");

if (configured(process.env.AUTH_GITHUB_ID) && configured(process.env.AUTH_GITHUB_SECRET)) {
  providers.push(GitHub);
}
if (configured(process.env.AUTH_GOOGLE_ID) && configured(process.env.AUTH_GOOGLE_SECRET)) {
  providers.push(Google);
}

/** Names of the OAuth providers that have credentials configured. */
export const configuredProviders = providers.map((p) =>
  typeof p === "function" ? p().id : p.id,
);

export const { handlers, auth, signIn, signOut } = NextAuth({
  adapter: PrismaAdapter(prisma),
  session: { strategy: "database" },
  providers,
  pages: { signIn: "/login" },
  trustHost: true,
});
