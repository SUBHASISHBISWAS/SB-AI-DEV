using System.ComponentModel;

namespace _1_FunctionCalls;

// 1. Define the Enterprise Tool
public static class LogisticsTools
{
    [Description("Retrieves the current shipping status of an enterprise logistics order. Invoke this tool ONLY when the user explicitly provides an Order ID.")]
    public static string GetOrderStatus(
        [Description("The exact, case-sensitive alphanumeric order identifier. Format must be 'ORD-' followed by 5 digits (e.g., ORD-12345).")] string orderId)
    {
        // Simulating a deterministic database or external API call
        if (orderId == "ORD-12345") return "IN TRANSIT - Estimated Delivery Tomorrow";
        if (orderId == "ORD-99999") return "PENDING - Awaiting Stock Validation";
        return "UNKNOWN - Order ID not found in the logistics system.";
    }
}