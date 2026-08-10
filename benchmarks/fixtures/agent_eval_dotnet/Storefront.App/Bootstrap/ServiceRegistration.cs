using Microsoft.Extensions.DependencyInjection;
using Storefront.App.Api;
using Storefront.App.Configuration;
using Storefront.App.Contracts;
using Storefront.App.Infrastructure;
using Storefront.App.Services;

namespace Storefront.App.Bootstrap;

public static class ServiceRegistration
{
    public static IServiceCollection AddStorefront(
        this IServiceCollection services)
    {
        services.AddSingleton<IOrderRepository, InMemoryOrderRepository>();
        services.AddTransient<OrderService>();
        services.AddTransient<OrderSummaryEndpoint>();
        services.AddTransient<OrderGateway>();
        services.Configure<RetryOptions>(options =>
        {
            options.MaxAttempts = 2;
            options.BaseDelayMilliseconds = 100;
        });
        return services;
    }
}
