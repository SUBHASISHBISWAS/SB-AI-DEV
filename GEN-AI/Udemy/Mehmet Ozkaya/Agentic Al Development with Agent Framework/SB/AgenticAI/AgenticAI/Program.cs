
using System.ClientModel;
using Azure.AI. OpenAI;
using Azure. Identity;
using Microsoft. Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI.Chat;

var endpoint=Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT")?? throw new InvalidOperationException();
var model = Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT_NAME") ?? "gpt-4o";

var chatClient = new AzureOpenAIClient(new Uri(endpoint), new AzureCliCredential()).GetChatClient(model)
    .AsIChatClient();

AIAgent supportAgent = chatClient.AsAIAgent(
    name: "NetworkSupport",
    instructions: "You are a Tier 1 IT Support Agent. Your answers must be concise, professional, and limited strict"
) ;
var userIssue = "I am getting a DNS resolution error when connecting to the corporate VPN from a coffee shop.";
Console. WriteLine($"User: {userIssue}\n");

var response= await supportAgent.RunAsync(userIssue);
Console.WriteLine($"Agent: {response.Text}");