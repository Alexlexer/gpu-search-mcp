namespace Storefront.App.Time;

public interface IClock
{
    DateTimeOffset UtcNow { get; }
}
