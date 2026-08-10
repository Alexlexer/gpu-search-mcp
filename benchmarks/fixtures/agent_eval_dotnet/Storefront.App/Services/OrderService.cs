using Storefront.App.Contracts;
using Storefront.App.Domain;

namespace Storefront.App.Services;

public sealed class OrderService(IOrderRepository repository)
{
    public Task<IReadOnlyList<Order>> GetCustomerOrdersAsync(
        string customerId,
        CancellationToken cancellationToken)
    {
        return repository.GetByCustomerAsync(
            customerId,
            CancellationToken.None);
    }
}
