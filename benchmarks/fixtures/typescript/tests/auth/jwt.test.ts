import { validateExpiration } from "../../src/auth/jwt";

test("rejects expired tokens", () => {
  expect(validateExpiration(10, 11)).toBe(false);
});
