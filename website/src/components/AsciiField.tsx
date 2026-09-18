"use client";

import { useEffect, useRef } from "react";

/**
 * Dot-matrix / character halftone field, rendered on a canvas.
 * Brightness follows a slow moving pseudo-noise so it reads as a wireframe
 * surface made of glyphs, like the halftone panels in the reference.
 */
export function AsciiField({ invert = false }: { invert?: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const glyphs = " .:-=+*#%@";
    let raf = 0;
    let w = 0;
    let h = 0;
    const cell = 11;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const io = new IntersectionObserver((e) => {
      visible = e[0]?.isIntersecting ?? true;
    });
    let visible = true;
    io.observe(canvas);

    const resize = () => {
      const r = canvas.getBoundingClientRect();
      w = Math.floor(r.width);
      h = Math.floor(r.height);
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(canvas);

    const draw = (t: number) => {
      raf = requestAnimationFrame(draw);
      if (!visible) return;
      const time = t * 0.00018;
      ctx.clearRect(0, 0, w, h);
      ctx.font = `10px var(--font-fragment), monospace`;
      ctx.textBaseline = "top";
      const cols = Math.ceil(w / cell);
      const rows = Math.ceil(h / cell);
      for (let y = 0; y < rows; y++) {
        for (let x = 0; x < cols; x++) {
          const nx = x / cols;
          const ny = y / rows;
          // ridge-like surface: two rotated sine sheets + a radial falloff
          const v1 = Math.sin(nx * 7 + ny * 3.2 + time * 2.1);
          const v2 = Math.sin(nx * 2.4 - ny * 8.1 - time * 1.4);
          const v3 = Math.sin((nx - 0.6) * (nx - 0.6) * 22 + (ny - 0.7) * (ny - 0.7) * 30 - time * 3);
          let v = (v1 + v2 + v3) / 3; // -1..1
          v = (v + 1) / 2;
          // keep most of the top-left empty like the reference
          const mask = Math.min(1, Math.max(0, (nx * 0.75 + ny * 1.1 - 0.3) * 1.7));
          const b = Math.pow(v, 1.25) * mask;
          if (b < 0.06) continue;
          const g = glyphs[Math.min(glyphs.length - 1, Math.floor(b * glyphs.length))];
          const shade = Math.floor(55 + b * 200);
          ctx.fillStyle = invert
            ? `rgb(${255 - shade},${255 - shade},${255 - shade})`
            : `rgb(${shade},${shade},${shade})`;
          ctx.fillText(g, x * cell, y * cell);
        }
      }
    };
    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
    };
  }, [invert]);

  return <canvas ref={ref} aria-hidden="true" />;
}
