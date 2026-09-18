export function Logo({ size = 22 }: { size?: number }) {
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 10,
        fontWeight: 500,
        letterSpacing: "-0.03em",
        fontSize: size,
        lineHeight: 1,
      }}
    >
      <svg
        width={size * 0.95}
        height={size * 0.95}
        viewBox="0 0 24 24"
        fill="none"
        aria-hidden="true"
      >
        <path d="M2 3h20L2 21V3Z" fill="currentColor" />
        <path d="M22 3v18H4" stroke="currentColor" strokeWidth="1.2" opacity="0.5" />
      </svg>
      karma
    </span>
  );
}
