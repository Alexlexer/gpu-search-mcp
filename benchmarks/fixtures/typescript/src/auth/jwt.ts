export function validateExpiration(expiresAt: number, now: number): boolean {
  return expiresAt > now;
}
