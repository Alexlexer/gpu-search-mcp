using Microsoft.Extensions.Options;
using Storefront.App.Configuration;
using Storefront.App.Time;

namespace Storefront.App.Infrastructure;

public sealed class OrderGateway(
    IOptions<RetryOptions> retryOptions,
    IClock clock)
{
    public int MaxAttempts => retryOptions.Value.MaxAttempts;

    public int BaseDelayMilliseconds => retryOptions.Value.BaseDelayMilliseconds;

    public DateTimeOffset GetAttemptTimestamp() => clock.UtcNow;
}
