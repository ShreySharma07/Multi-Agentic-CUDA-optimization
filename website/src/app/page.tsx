import { SiteNav } from "@/components/SiteNav";
import { Footer } from "@/components/Footer";
import { Hero } from "@/components/Hero";
import { Marquee } from "@/components/Marquee";
import { Loop } from "@/components/Loop";
import { HowItWorks } from "@/components/HowItWorks";
import { Architecture } from "@/components/Architecture";
import { Results } from "@/components/Results";
import { Desktop } from "@/components/Desktop";
import { Cta } from "@/components/Cta";

export default function Home() {
  return (
    <>
      <SiteNav />
      <main>
        <Hero />
        <Marquee />
        <Loop />
        <HowItWorks />
        <Architecture />
        <Results />
        <Desktop />
        <Cta />
      </main>
      <Footer />
    </>
  );
}
