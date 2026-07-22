import { validateExpiration } from "./jwt";

export function acceptToken(expiresAt: number, now: number): boolean {
  return validateExpiration(expiresAt, now);
}
