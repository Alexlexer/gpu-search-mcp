namespace Storefront.App.Domain;

public enum OrderStatus
{
    Pending,
    Processing,
    Completed,
    Cancelled,
}

public sealed record Order(
    string Id,
    string CustomerId,
    string ExternalReference,
    OrderStatus Status,
    DateTimeOffset CreatedAt);
