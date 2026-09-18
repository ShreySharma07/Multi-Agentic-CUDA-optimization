# KARMA website

Marketing site, OAuth login and desktop early-access waitlist for KARMA, the
multi-agent CUDA kernel optimizer in the parent repository.

Stack: Next.js 16 (App Router, Turbopack), React 19, Three.js via
`@react-three/fiber` for the GPU scene, Auth.js v5 with the Prisma adapter,
Prisma 6 on Postgres, plain CSS modules. No UI framework.

## Run locally

Needs a Postgres database (Homebrew `postgresql@15` works; Neon's free tier
works too).

```bash
cd website
npm install
cp .env.example .env          # set DATABASE_URL, AUTH_SECRET, OAuth keys
createdb karma                # if using local Postgres
npx prisma migrate deploy     # applies prisma/migrations
npm run dev                   # http://localhost:3000
```

## Deploy to Vercel

Vercel is the target: Next.js 16 server actions and the auth route run as
serverless functions, and the build script runs the Prisma migration.

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2FShreySharma07%2FMulti-Agentic-CUDA-optimization&root-directory=website&project-name=karma&env=AUTH_SECRET,AUTH_GITHUB_ID,AUTH_GITHUB_SECRET,AUTH_GOOGLE_ID,AUTH_GOOGLE_SECRET&envDescription=Auth.js%20secret%20and%20OAuth%20app%20credentials&stores=%5B%7B%22type%22%3A%22postgres%22%7D%5D)

Or by hand:

1. Vercel dashboard → Add New → Project → import this GitHub repo.
2. **Root Directory**: `website`. Framework is detected as Next.js. Keep the
   default build command (`npm run build` runs `prisma generate`,
   `prisma migrate deploy`, `next build`).
3. **Storage** tab → Create Database → Neon Postgres (or any Postgres) and
   connect it to the project. This injects `DATABASE_URL`.
4. **Environment Variables**: `AUTH_SECRET` (`openssl rand -base64 32`),
   `AUTH_GITHUB_ID`, `AUTH_GITHUB_SECRET`, `AUTH_GOOGLE_ID`,
   `AUTH_GOOGLE_SECRET`. `AUTH_TRUST_HOST` is not needed on Vercel.
5. Deploy. Then set each OAuth app's callback URL to
   `https://<your-domain>/api/auth/callback/github` and
   `.../callback/google`.

From the CLI instead: `npm i -g vercel && vercel login && vercel link`
(root `website`), then `vercel env add` for the variables above and
`vercel --prod`.

Render also works (Web Service, root `website`, build `npm install && npm run
build`, start `npm start`, plus a Render Postgres instance for
`DATABASE_URL`), but its free tier sleeps between requests and cold-starts
the Three.js pages slowly, so Vercel is the better fit.

## OAuth

Sign-in uses GitHub and/or Google. A provider only appears on `/login` when
both of its variables are set in `.env`:

| Provider | Variables                               | Callback URL                                  |
| -------- | --------------------------------------- | --------------------------------------------- |
| GitHub   | `AUTH_GITHUB_ID`, `AUTH_GITHUB_SECRET`  | `http://localhost:3000/api/auth/callback/github` |
| Google   | `AUTH_GOOGLE_ID`, `AUTH_GOOGLE_SECRET`  | `http://localhost:3000/api/auth/callback/google` |

Generate `AUTH_SECRET` with `openssl rand -base64 32`. In production set the
callback URLs to your domain. `AUTH_TRUST_HOST=true` is only needed behind a
non-Vercel proxy such as Render.

## Early access

`/early-access` writes to the `EarlyAccess` table (email, GPU, platform, use
case, notes). Signed-in users get the row linked to their account and can see
their queue position on `/dashboard`. Anonymous requests are keyed by email.

Inspect the waitlist with `npx prisma studio`.

## Where things live

- `src/app/page.tsx` — landing page composition
- `src/components/GpuScene.tsx` — Three.js RTX A4000 (8 × 6 SM die) with the
  five scroll-driven stages; `STAGES` holds the copy per stage
- `src/components/HowItWorks.tsx` — scroll-pinned section that drives the scene
- `src/lib/results.ts` — measured numbers copied from `../results/experiments.csv`
- `src/auth.ts` — Auth.js config; `prisma/schema.prisma` — data model;
  `prisma/migrations/` — SQL applied by `prisma migrate deploy` on every build
