"use server";

import { z } from "zod";
import { auth } from "@/auth";
import { prisma } from "@/lib/prisma";

export type EarlyAccessState =
  | { status: "idle" }
  | { status: "error"; message: string; fields?: Record<string, string>; values: Record<string, string> }
  | { status: "ok"; position: number; email: string };

const Schema = z.object({
  email: z.string().trim().toLowerCase().email("Enter a valid email."),
  gpu: z.string().trim().min(1, "Pick your GPU.").max(80),
  platform: z.enum(["linux", "windows", "macos"], { message: "Pick a platform." }),
  useCase: z.enum(["inference", "training", "research", "hpc", "other"], { message: "Pick a use case." }),
  notes: z.string().trim().max(600, "Keep notes under 600 characters.").optional().or(z.literal("")),
});

export async function requestEarlyAccess(
  _prev: EarlyAccessState,
  formData: FormData,
): Promise<EarlyAccessState> {
  const raw = {
    email: String(formData.get("email") ?? ""),
    gpu: String(formData.get("gpu") ?? ""),
    platform: String(formData.get("platform") ?? ""),
    useCase: String(formData.get("useCase") ?? ""),
    notes: String(formData.get("notes") ?? ""),
  };
  const parsed = Schema.safeParse(raw);

  if (!parsed.success) {
    const fields: Record<string, string> = {};
    for (const issue of parsed.error.issues) {
      const k = String(issue.path[0] ?? "form");
      if (!fields[k]) fields[k] = issue.message;
    }
    return { status: "error", message: "Please fix the highlighted fields.", fields, values: raw };
  }

  const session = await auth();
  const userId = session?.user?.id ?? null;
  const data = {
    email: parsed.data.email,
    gpu: parsed.data.gpu,
    platform: parsed.data.platform,
    useCase: parsed.data.useCase,
    notes: parsed.data.notes || null,
  };

  try {
    // Prefer the signed-in user's row; otherwise key on email.
    const existing = userId
      ? await prisma.earlyAccess.findFirst({ where: { OR: [{ userId }, { email: data.email }] } })
      : await prisma.earlyAccess.findUnique({ where: { email: data.email } });

    const row = existing
      ? await prisma.earlyAccess.update({
          where: { id: existing.id },
          data: { ...data, userId: userId ?? existing.userId },
        })
      : await prisma.earlyAccess.create({ data: { ...data, userId } });

    const position = await prisma.earlyAccess.count({ where: { createdAt: { lte: row.createdAt } } });
    return { status: "ok", position, email: row.email };
  } catch (e) {
    console.error("early access", e);
    return { status: "error", message: "Could not save your request. Please try again.", values: raw };
  }
}
