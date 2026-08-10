using Storefront.App.Domain;
using Storefront.App.Services;

namespace Storefront.App.Api;

public sealed record OrderListItem(
    string Id,
    string ExternalReference,
    OrderStatus Status,
    DateTimeOffset CreatedAt);

public sealed record OrderSummaryResponse(
    IReadOnlyList<OrderListItem> Orders,
    int Total);

public sealed class OrderSummaryEndpoint(OrderService orderService)
{
    public async Task<OrderSummaryResponse> GetOpenOrdersAsync(
        string customerId,
        CancellationToken cancellationToken)
    {
        var orders = await orderService.GetCustomerOrdersAsync(
            customerId,
            CancellationToken.None);

        var items = orders
            .Where(order => order.Status != OrderStatus.Cancelled)
            .OrderBy(order => order.CreatedAt)
            .Select(order => new OrderListItem(
                order.Id,
                order.ExternalReference,
                order.Status,
                order.CreatedAt))
            .ToArray();

        return new OrderSummaryResponse(items, items.Length);
    }
}
