using Storefront.App.Contracts;
using Storefront.App.Domain;

namespace Storefront.App.Infrastructure;

public sealed class InMemoryOrderRepository : IOrderRepository
{
    private readonly IReadOnlyList<Order> _orders;

    public InMemoryOrderRepository()
        : this(Array.Empty<Order>())
    {
    }

    public InMemoryOrderRepository(IEnumerable<Order> orders)
    {
        _orders = orders.ToArray();
    }

    public Task<IReadOnlyList<Order>> GetByCustomerAsync(
        string customerId,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        IReadOnlyList<Order> matches = _orders
            .Where(order => string.Equals(
                order.CustomerId,
                customerId,
                StringComparison.Ordinal))
            .ToArray();
        return Task.FromResult(matches);
    }
}
