using Storefront.App.Domain;

namespace Storefront.App.Contracts;

public interface IOrderRepository
{
    Task<IReadOnlyList<Order>> GetByCustomerAsync(
        string customerId,
        CancellationToken cancellationToken);
}
