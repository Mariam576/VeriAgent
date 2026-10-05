// Small helpers for showing values. The numbers themselves always come from the server.

const SYMBOLS: Record<string, string> = { GBP: "£", USD: "$", EUR: "€", INR: "₹", JPY: "¥" };

export function formatNumber(value: number): string {
  return Number.isInteger(value)
    ? value.toLocaleString("en-GB")
    : value.toLocaleString("en-GB", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

/** 58000, "GBP" -> "£58,000";  30, "days" -> "30 days";  44, null -> "44" */
export function formatValue(value: number | null | undefined, unit: string | null | undefined): string {
  if (value === null || value === undefined) return "";
  const number = formatNumber(Math.abs(value));
  const sign = value < 0 ? "-" : "";
  if (!unit) return sign + number;
  const code = unit.toUpperCase();
  if (SYMBOLS[code]) return `${sign}${SYMBOLS[code]}${number}`;
  if (unit === "%") return `${sign}${number}%`;
  if (/^[A-Z]{3}$/.test(unit)) return `${unit} ${sign}${number}`;
  return `${sign}${number} ${unit}`;
}

export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return "";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}

export function percent(fraction: number): string {
  return `${Math.round(fraction * 100)}%`;
}

export function titleCase(text: string): string {
  const spaced = text.replace(/_/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

export function errorText(error: unknown): string {
  return error instanceof Error ? error.message : "Something went wrong.";
}

/** Only links to GitHub issues are ever turned into clickable links. */
export function safeLink(url: string | null): string | null {
  return url && /^https:\/\/github\.com\/[\w.-]+\/[\w.-]+\/issues\/\d+$/.test(url) ? url : null;
}
