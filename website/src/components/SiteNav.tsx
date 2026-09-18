import { auth } from "@/auth";
import { Nav } from "./Nav";

export async function SiteNav() {
  const session = await auth();
  return <Nav user={session?.user ?? null} />;
}
