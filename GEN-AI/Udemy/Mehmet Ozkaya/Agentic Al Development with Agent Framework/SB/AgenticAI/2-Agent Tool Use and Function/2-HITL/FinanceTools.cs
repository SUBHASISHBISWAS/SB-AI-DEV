using System.ComponentModel;

namespace _2_HITL;

// 1. Define the Sensitive Enterprise Tool
public static class FinanceTools
{
    [Description("Issues a financial refund to a customer. Use this ONLY when the user explicitly requests a refund and provides an Order ID.")]
    public static string IssueRefund(
        [Description("The Order ID to refund (e.g., ORD-12345).")] string orderId,
        [Description("The decimal amount to refund.")] decimal amount)
    {
        // Simulating a deterministic call to a payment gateway (e.g., Stripe or PayPal)
        Console.WriteLine($"\n[SYSTEM LOG] Executing secure transaction: Refunded ${amount} to {orderId}.\n");
        return $"SUCCESS: ${amount} has been refunded to order {orderId}.";
    }
}