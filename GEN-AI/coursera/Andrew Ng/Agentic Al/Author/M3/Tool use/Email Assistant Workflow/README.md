# 🤖 Tools-in-Agents — Email Management Demo

This repository is part of a hands-on course from **DeepLearning.AI**, designed to teach you how large language models (LLMs) can be equipped with **tools** to perform complex, real-world tasks like managing an inbox. It’s not just about generating text — it’s about **thinking and acting**.

You’ll build and observe an LLM-based agent that:
- Thinks about what it needs to do
- Calls Python tools (functions or API endpoints)
- Uses the results to plan the next step

All using a ReAct-style loop: **Reason → Act → Observe → Repeat**

---

## 🚀 Getting Started

### ⚙️ 1. Install Requirements

Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 🔐 2. Add Your API Keys

Create a file called `.env` in the root of the repository with your API keys:

```env
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
TAVILY_API_KEY=tvly-...
```

These are used by the agent backend to route prompts to supported models via aisuite.

### ▶️ 3. Start the Backend Services

**In one terminal**, start the email API server:

```bash
cd M3/email_agent/email_server
uvicorn email_service:app --timeout-keep-alive 1200
```

📍 This runs on: `http://localhost:8000`

It loads a mock inbox with 5 sample emails (all unread).

**In another terminal**, start the LLM agent server:

```bash
cd M3/email_agent
uvicorn llm_service:app --port 8001 --timeout-keep-alive 1200
```

📍 This runs on: `http://localhost:8001/prompt`

**To run the frontend UI**, simply open `ui_all.html` in a browser, otherwise, go to next step to work with the notebook.

### 💡 4. Open the Notebook

Start Jupyter:

```bash
jupyter lab
```

Open `agent_tools_email_service.ipynb` and follow the instructions. The notebook will:
- Explain how tools are registered
- Let you send a prompt
- Show how the LLM reasons and acts step-by-step

---

## 📦 Project Structure Explained

```
EMAIL-AGENT/
├── email_server/
│   ├── email_service.py      # FastAPI backend with REST endpoints
│   ├── email_models.py       # SQLAlchemy models for Email objects
│   ├── email_schema.py       # Pydantic validation classes
│   └── email_database.py     # SQLite setup + preload logic (inserts 5 sample emails)
├── llm_service.py            # LLM interface (FastAPI + aisuite tool agent)
├── email_tools.py            # Python tools the agent can call (list, search, send)
├── agent_tools_email_service.ipynb  # Pedagogical notebook with live examples
├── display_functions.py      # Pretty-printing utilities for chat completions
├── ui_all.html               # (Optional) visual interface for browsing emails
├── utils.py                  # Utility functions (e.g., timestamp formatting)
├── requirements.txt          # Project dependencies
├── .env                      # API keys file (not committed)
└── README.md                 # You are here ✅
```

---

## 🧠 Key Concept: Tools in Agents

Tools are Python functions that the LLM can invoke when it needs to **act**.

For example, given this prompt:

> "Check for unread emails from boss@email.com and reply politely."

The agent may:
1. Call `list_unread_emails()`
2. Filter for sender
3. Call `mark_email_as_read(email_id=...)`
4. Call `send_email(...)`

The entire decision chain is made by the LLM, not hardcoded. You can inspect every step.

---

## 🔧 Tools Available to the Agent

These are defined in `email_tools.py`:

```python
list_unread_emails()
search_emails(query: str)
get_email(email_id: int)
mark_email_as_read(email_id: int)
send_email(recipient: str, subject: str, body: str)
search_unread_from_sender(sender: str)
```

They are passed to the agent via:

```python
client.chat.completions.create(
    model="openai:o4-mini",
    messages=[{"role": "user", "content": prompt}],
    tools=[...],
    max_turns=5
)
```

---

## 📡 Backend API Reference

The email system is backed by FastAPI and offers these endpoints:

| Method   | Route                       | Description                     |
|----------|----------------------------|---------------------------------|
| `GET`    | `/reset_database`          | Reloads the 5 default emails    |
| `POST`   | `/send`                    | Sends a mock email              |
| `GET`    | `/emails`                  | Lists all emails                |
| `GET`    | `/emails/unread`           | Lists unread emails             |
| `GET`    | `/emails/search?q=...`     | Search by subject/body/sender   |
| `GET`    | `/emails/filter`           | Filter by recipient or date     |
| `GET`    | `/emails/{email_id}`       | Get email by ID                 |
| `PATCH`  | `/emails/{email_id}/read`  | Mark as read                    |
| `PATCH`  | `/emails/{email_id}/unread`| Mark as unread                  |
| `DELETE` | `/emails/{email_id}`       | Delete email                    |

---

## 🧪 Try This Prompt

Paste this into the notebook and watch the tools fire:

```text
Check for unread emails from boss@email.com,
mark them as read,
and send a polite follow-up.
```

You'll see:
- Tool calls executed
- Responses interpreted
- Final message generated

All automatically, via agent reasoning.

---

## ✅ Summary

This project is designed for learners. You’ll:
- Build and run your own agent system
- Watch LLMs call functions, not just generate text
- Understand how reasoning and actions can be chained
- Use `aisuite` to connect prompts to tool execution
