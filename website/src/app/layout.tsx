import type { Metadata } from "next";
import { Host_Grotesk, Fragment_Mono } from "next/font/google";
import "./globals.css";

const host = Host_Grotesk({
  variable: "--font-host",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600"],
  display: "swap",
});

const fragment = Fragment_Mono({
  variable: "--font-fragment",
  subsets: ["latin"],
  weight: "400",
  display: "swap",
});

export const metadata: Metadata = {
  title: "KARMA — kernels that optimize themselves",
  description:
    "KARMA is a closed-loop, multi-agent system that profiles, rewrites, compiles, validates and benchmarks CUDA kernels on real hardware, and ships only measured, correct speedups.",
  metadataBase: new URL(
    process.env.NEXT_PUBLIC_SITE_URL ??
      (process.env.VERCEL_PROJECT_PRODUCTION_URL
        ? `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}`
        : "http://localhost:3000"),
  ),
  openGraph: {
    title: "KARMA — kernels that optimize themselves",
    description:
      "Profile. Rewrite. Compile. Validate. Benchmark. Reflect. A multi-agent loop for CUDA kernels.",
    type: "website",
  },
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${host.variable} ${fragment.variable}`}>
      <body>{children}</body>
    </html>
  );
}
