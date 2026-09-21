using A2A.AspNetCore;
using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Hosting;
using Microsoft.Agents.AI.Hosting.A2A;
using Microsoft.Extensions.AI;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.

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

// Register the Specialized Enterprise Agent
var complianceAgent = builder.AddAIAgent(
    name: "compliance",
    instructions: "You are a strict enterprise compliance auditor. Review the provided text for GDPR violations. Be concise and authoritative."
);

// Register A2A server for the agent (required by latest A2A endpoint mapping APIs)
complianceAgent.AddA2AServer();

var app = builder.Build();

// Configure the HTTP request pipeline.

// Expose the agent via the A2A HTTP+JSON protocol
app.MapA2AHttpJson(complianceAgent, path: "/a2a/compliance");

app.Run();

