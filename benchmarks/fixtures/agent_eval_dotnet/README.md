# Storefront evaluation fixture

This small .NET storefront models production-style boundaries used by the coding-agent
evaluation suite: pricing, repositories, services, endpoint shaping, dependency
injection, configuration, and cancellation.

The fixture intentionally contains independent defects and missing behavior. Evaluation
tasks start from one immutable Git commit and validate one requested change at a time.
It has no external NuGet package dependency; the application uses the ASP.NET Core
shared framework included with the .NET SDK.
