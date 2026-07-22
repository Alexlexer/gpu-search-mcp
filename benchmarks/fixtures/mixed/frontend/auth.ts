export const AUTH_TTL_SECONDS = 3600;

export function tokenIsFresh(ageSeconds: number): boolean {
  return ageSeconds < AUTH_TTL_SECONDS;
}
