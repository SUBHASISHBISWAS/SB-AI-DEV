# Agentic AI Development with Agent Framework, MCP and .NET

This repository contains the source code used in my Udemy course:

**Agentic AI Development with Agent Framework, MCP and .NET**  
https://www.udemy.com/course/agent-development-microsoft-agent-framework-mcp-and-net/?couponCode=LAUNCH

Welcome to the definitive guide on building production-ready agentic AI systems in the .NET ecosystem. This course goes beyond theory and focuses on hands-on development of autonomous multi-agent orchestration for enterprise applications.

Using **Microsoft Agent Framework**, **Microsoft Foundry**, the **Model Context Protocol (MCP)**, **.NET Aspire**, **AG-UI**, **DevUI**, and **.NET**, you will learn how to build robust AI workflows that solve complex business problems.

---

## What You Will Master

- **Microsoft Agent Framework (MAF)**: Build sophisticated, stateful AI systems with Microsoft Foundry and Azure OpenAI.
- **Multi-Agent Orchestration**: Implement Sequential, Concurrent, Handoff, and Group Chat patterns.
- **Agentic RAG Systems**: Build intent-based retrieval using Qdrant vector databases and `TextSearchProvider`.
- **Protocols & Interoperability**: Implement A2A communication, MCP tool exposure, and AG-UI frontend streaming.
- **Observability & Visual Testing**: Track JSON payloads, token usage, handoff latencies, and tool calls with Aspire + DevUI.
- **Enterprise Microservices Integration**: Integrate AI agents into MinimalAgent-style microservices and existing APIs.

---

## Course Roadmap

### Part 1: Core Agent Development

Learn core agent anatomy, Azure OpenAI connectivity, invocation lifecycle, custom function tools, and persistent conversational context with `AgentSession`.

### Part 2: Orchestrating Multi-Agent Systems

Design and implement specialized swarms (triage, finance, compliance) with `AgentWorkflowBuilder` and production workflow topologies.

### Part 3: Advanced Reasoning with Agentic RAG

Build systems where agents decide when retrieval is needed. Integrate Qdrant, embeddings, and semantic search into enterprise-ready AI flows.

### Part 4: Agent Communications and Protocols

Expose agents through Web APIs for A2A architectures, connect to local/hosted MCP servers, and stream generative UI experiences with AG-UI.

---

## Repository Structure (`src`)

The repository is organized by section, matching the course progression:

- `section4-setup`
  - `HelloAgent`
- `section5-getting-started`
  - `BasicAgentApp`, `MultiTurn`, `Streaming`, `StructuredOutput`
  - `MinimalAgent` (Aspire-style AppHost/WebApi/ServiceDefaults)
- `section6-tool-use`
  - `FunctionCall`, `CodeInterpreter`, `ApproveRequiredFunc`
  - `MinimalAgentWithTools` (AppHost/WebApi/ServiceDefaults)
- `section7-memory`
  - `CustomContextProvider`, `MockCosmosDb`
- `section8-workflows`
  - `FirstWorkflow`, `AgentsWorkflow`
- `section9-workflow-patterns`
  - `AgenticWorkflowPatterns`
  - `MinimalAgentWithWorkflows` (AppHost/WebApi/ServiceDefaults)
- `section11-agentic-rag`
  - `BasicTextRAGExample`, `QdrantVectorStore`
- `section13-a2a`
  - `EnterpriseComplianceService`
- `section14-mcp`
  - `HostedMcpGovernanceExample`, `LocalGitHubMcpExample`
- `section15-ag-ui`
  - `Server`, `Client`

Solution file:

- `src/develop-agents.slnx`

---

## Technology Stack

- **Languages & Frameworks**: .NET 10, C#, ASP.NET Core, Blazor Server
- **AI & Agents**: Microsoft Agent Framework, Azure OpenAI (`gpt-5-mini`, `text-embedding-3-small`)
- **Cloud & Deployment**: Microsoft AI Foundry
- **Frontend Protocol**: AG-UI
- **Orchestration**: .NET Aspire
- **Observability**: Aspire OpenTelemetry, Application Insights
- **Vector Data & Storage**: Qdrant
- **Architecture**: Microservices, Clean Architecture, MCP

---

## Who This Course Is For

- Software Developers, Architects, and Tech Leads in enterprise environments transitioning into AI Engineering.
- Professionals integrating AI agents into backend microservices, legacy systems, and APIs.
- C#/.NET developers who want to build agentic AI systems without switching to Python.
