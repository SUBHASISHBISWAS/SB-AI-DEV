using Azure.AI.OpenAI;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using OpenAI.Assistants;
using OpenAI.Chat;
using OpenAI.Responses;


// Define the variables we extracted from Microsoft Foundry
var endpoint = Environment.GetEnvironmentVariable("AZURE_OPENAI_ENDPOINT") ?? throw new InvalidOperationException("AZURE_OPENAI_ENDPOINT is not set.");
var deploymentName = Environment.GetEnvironmentVariable("AZURE_OPENAI_DEPLOYMENT_NAME") ?? "gpt-5-mini";

// Build the tools list and add the native Code Interpreter ResponseTool via the AITool bridge extension
List<AITool> tools = [];
#pragma warning disable OPENAI001
tools.Add(ResponseTool.CreateCodeInterpreterTool(
    new CodeInterpreterToolContainer(CodeInterpreterToolContainerConfiguration.CreateAutomaticContainerConfiguration([]))));
#pragma warning restore OPENAI001

// Initialize the Agent and inject the native Code Interpreter tool
AIAgent mathAgent = new AzureOpenAIClient(new Uri(endpoint), new AzureCliCredential())
    .GetChatClient(deploymentName)
    .AsAIAgent(
        name: "DataAnalyst",
        instructions: "You are a data analyst. You must write and execute Python code to answer complex math and data questions. Never guess the answer.",
        tools: tools
    );

Console.WriteLine($"Agent '{mathAgent.Name}' is online with a Python sandbox.\n");

// The agent will autonomously write a script to solve this, run it, and return the result.
string prompt = "Calculate the 10th Fibonacci number and determine if it is a prime number.";
Console.WriteLine($"User: {prompt}");

AgentResponse response = await mathAgent.RunAsync(prompt);
Console.WriteLine($"Agent: {response.Text}");


//// File Search Tool
//// In a real scenario, this ID is retrieved from your Azure AI Foundry project
//string corporateVectorStoreId = "vs-987654321";

//// Initialize the Agent with the File Search capability pointing to your data
//AIAgent financeAgent = new AzureOpenAIClient(new Uri(endpoint), new AzureCliCredential())
//    .GetChatClient(deploymentName)
//    .AsAIAgent(
//        name: "FinancialAnalyst",
//        instructions: "You are a financial analyst. Answer questions strictly based on the provided corporate documents...",
//        tools: [new FileSearchToolDefinition(vectorStoreIds: [corporateVectorStoreId])]
//    );

//string prompt = "What were the key risk factors in the Q3 Report?";

//// Web Search tool
//// Initialize the Agent with live internet access
//AIAgent researchAgent = new AzureOpenAIClient(new Uri(endpoint), new AzureCliCredential())
//    .GetChatClient(deploymentName)
//    .AsAIAgent(
//        name: "MarketResearcher",
//        instructions: "You are a market researcher. Always verify current events using the Web Search tool before providing an answer. Cite your sources.",
//        tools: [new WebSearchToolDefinition()]
//    );

//string prompt = "What were the yesterday's major tech announcements";