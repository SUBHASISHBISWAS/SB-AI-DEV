from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import aisuite as ai
from dotenv import load_dotenv
from .display_functions import pretty_print_chat_completion_html
import markdown

# Importa las herramientas decoradas con @tool
from .email_tools import (
    list_all_emails,
    list_unread_emails,
    search_emails,
    filter_emails,
    get_email,
    mark_email_as_read,
    mark_email_as_unread,
    send_email,
    delete_email,
    search_unread_from_sender
)

load_dotenv()
client = ai.Client()

app = FastAPI(title="LLM Email Prompt Executor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptInput(BaseModel):
    prompt: str

@app.post("/prompt")
async def handle_prompt(payload: PromptInput):
    prompt = payload.prompt

    prompt_ = f"""
        - You are an AI assistant specialized in managing emails. 
        - You can perform various actions such as listing, searching, filtering, and manipulating emails. 
        - Use the provided tools to interact with the email system.
        - Never ask the user for confirmation before performing an action.
        - If needed, my email address is "you@email.com" so  you can use it to send emails or perform actions related to my account.
        {prompt}
        """

    response = client.chat.completions.create(
        model="openai:gpt-4.1",
        messages=[{"role": "user", "content": prompt_}],
        tools=[
            list_all_emails,
            list_unread_emails,
            search_emails,
            filter_emails,
            get_email,
            mark_email_as_read,
            mark_email_as_unread,
            send_email,
            delete_email,
            search_unread_from_sender
        ],
        max_turns=20
    )

    html_response = pretty_print_chat_completion_html(response)
    final_text = markdown.markdown(response.choices[0].message.content)

    return {
        "response": final_text,
        "html_response": html_response
    }
