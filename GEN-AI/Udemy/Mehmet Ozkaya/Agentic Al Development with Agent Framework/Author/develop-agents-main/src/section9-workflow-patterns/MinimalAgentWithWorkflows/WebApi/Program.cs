using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.DevUI;
using Microsoft.Agents.AI.Hosting;
using Microsoft.Agents.AI.Workflows;
using Microsoft.Extensions.AI;

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

/////// AGENT DEFINITIONS ///////

// Triage Agent - routes to specialists
builder.AddAIAgent("TriageAgent", (sp, name) =>
{
    var chat = sp.GetRequiredService<IChatClient>();
    return new ChatClientAgent(
        chatClient: chat,
        instructions:
            """
            You are a routing manager.
            Analyze the customer’s request and route to the right specialist.                        
            Do not attempt to answer domain questions yourself. You are only a router.
            Keep your response to 1-2 sentences.
            """,
        name: name,
        description: "Routes requests to the correct specialist agent.",
        tools: null,
        loggerFactory: sp.GetService<ILoggerFactory>(),
        services: sp);
});

// Order Agent - handles order requests
builder.AddAIAgent("OrderAgent", (sp, name) =>
{
    var chat = sp.GetRequiredService<IChatClient>();
    return new ChatClientAgent(
        chatClient: chat,
        instructions:
            """
            You are a logistics specialist. 
            You handle replacements, tracking, and shipping preferences.
            Keep your response to 1-2 sentences.
            """,
        name: name,
        description: "Handles order requests.",
        tools: null,
        loggerFactory: sp.GetService<ILoggerFactory>(),
        services: sp);
});

// Refund Agent - handles refund requests
builder.AddAIAgent("RefundAgent", (sp, name) =>
{
    var chat = sp.GetRequiredService<IChatClient>();
    return new ChatClientAgent(
        chatClient: chat,
        instructions:
            """
            You are a finance specialist. 
            You look up order details, gather context, and process refund requests.
            Keep your response to 1-2 sentences.
            """,
        name: name,
        description: "Handles refund requests.",
        tools: null,
        loggerFactory: sp.GetService<ILoggerFactory>(),
        services: sp);
});

/////// SEQUENTIAL WORKFLOW (registered as an agent for DevUI discovery) ///////
builder.AddWorkflow("OrderRefundWorkflow_Sequential", (sp, name) =>
{
    var triageAgent = sp.GetRequiredKeyedService<AIAgent>("TriageAgent");
    var orderAgent = sp.GetRequiredKeyedService<AIAgent>("OrderAgent");
    var refundAgent = sp.GetRequiredKeyedService<AIAgent>("RefundAgent");

    var workflow = AgentWorkflowBuilder
        .BuildSequential(name, triageAgent, orderAgent, refundAgent);    

    return workflow;
}).AddAsAIAgent();


/////// GROUP CHAT WORKFLOW (registered as an agent for DevUI discovery) ///////
builder.AddWorkflow("OrderRefundWorkflow_GroupChat", (sp, name) =>
{
    var triageAgent = sp.GetRequiredKeyedService<AIAgent>("TriageAgent");
    var orderAgent = sp.GetRequiredKeyedService<AIAgent>("OrderAgent");
    var refundAgent = sp.GetRequiredKeyedService<AIAgent>("RefundAgent");

    // Group chat
    var workflow = AgentWorkflowBuilder
        .CreateGroupChatBuilderWith(agents =>
            new RoundRobinGroupChatManager(agents)
            {
                MaximumIterationCount = 3  // One turn per agent (Triage, Order, Refund)
            })
        .AddParticipants(triageAgent, orderAgent, refundAgent)
        .WithName(name)
        .WithDescription("An order and refund workflow that routes requests to the correct specialist agent and handles multi-turn conversations until resolution.")
        .Build();

    return workflow;
}).AddAsAIAgent();


// 5. Register DevUI services
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

//// Map chat endpoint to trigger the agent
//app.MapPost("/api/chat", async (ChatRequest request,
//    [FromKeyedServices("NetworkSupportAgent")] AIAgent networkSupportAgent) =>
//{
//    var response = await networkSupportAgent.RunAsync(request.Message);
//    return Results.Ok(new { response = response.Text });
//});


app.Run();

record ChatRequest(string Message);
