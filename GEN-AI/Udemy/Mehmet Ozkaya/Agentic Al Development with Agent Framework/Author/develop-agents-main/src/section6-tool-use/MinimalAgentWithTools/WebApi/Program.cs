using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.DevUI;
using Microsoft.Agents.AI.Hosting;
using Microsoft.Extensions.AI;
using System.ComponentModel;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.AddServiceDefaults();

// 1. Define the variables we extracted from Microsoft Foundry
var endpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT") ?? throw new InvalidOperationException("AZURE_OPENAI_ENDPOINT is not set.");
var deploymentName = Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT_NAME") ?? "gpt-5-mini";

// 2. Instantiate the universal chat client with OpenTelemetry GenAI instrumentation
IChatClient chatClient = new AzureOpenAIClient(
        new Uri(endpoint),
        new AzureCliCredential())
    .GetChatClient(deploymentName)
    .AsIChatClient()
    .AsBuilder()
    .UseOpenTelemetry(configure: c => c.EnableSensitiveData = true)
    .Build();
builder.Services.AddSingleton(chatClient);

// 3. Define and Register the Agents
builder.AddAIAgent(
    name: "NetworkSupportAgent",
    instructions:
        """
        You are a Tier 1 IT Support Agent.
        Your answers must be concise, professional, and limited strictly to troubleshooting network and VPN connectivity. 
        Always check their VPN status if they report a disconnection.
        Keep responses concise — 3-5 sentences per turn. Be direct and opinionated.        
        """,
    chatClient)
    .WithAITools(AIFunctionFactory.Create(NetworkTools.CheckVpnStatus));

// 4. Register DevUI services
builder.AddDevUI();
builder.Services.AddOpenAIResponses();
builder.Services.AddOpenAIConversations();


var app = builder.Build();

// Configure the HTTP request pipeline.
app.MapDefaultEndpoints();

// Map DevUI endpoints 
app.MapDevUI();
app.MapOpenAIResponses();
app.MapOpenAIConversations();

// Map chat endpoint to trigger the agent
app.MapPost("/api/chat", async (ChatRequest request,
    [FromKeyedServices("NetworkSupportAgent")] AIAgent networkSupportAgent) =>
{
    var response = await networkSupportAgent.RunAsync(request.Message);
    return Results.Ok(new { response = response.Text });
});

app.Run();

record ChatRequest(string Message);

// 1. Define the Enterprise Tool
public static class NetworkTools
{
    [Description("Checks the current status of the corporate VPN for a specific user.")]
    public static string CheckVpnStatus(
        [Description("The username of the employee, e.g., jsmith")] string username)
    {
        // Simulating a deterministic API call to a firewall or directory service
        return $"User {username} is currently DISCONNECTED. Error: IPsec tunnel timeout.";
    }
}