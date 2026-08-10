namespace Storefront.App.Configuration;

public sealed class RetryOptions
{
    public const string SectionName = "Storefront:Retry";

    public int MaxAttempts { get; set; } = 2;

    public int BaseDelayMilliseconds { get; set; } = 100;
}
