# import dotenv

# dotenv.load_dotenv()

from pydantic import BaseModel, field_validator
import pandas as pd
import torch
import torch.nn as nn
import ast
import time
import os
import subprocess
import pathlib
import google
import asyncio
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.genai import types
from google.adk.sessions import InMemorySessionService


import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.ERROR)

PROFILE_PATH = "kernelpractise/sigmoid_profile.csv"
CUDA_PATH = "kernelpractise/sigmoid_kernel.cu"

try:
    if os.path.exists(CUDA_PATH):
        with open(CUDA_PATH, 'r') as f:
            cuda_code = f.read()
except Exception as e:
    print(e)

try:
    if os.path.exists(PROFILE_PATH):
        with open(PROFILE_PATH, 'r') as f:
            profile = f.read()
except Exception as e:
    print(e)

# os.environ['GOOGLE_API_KEY'] = 'GOOGLE_API'

AGENT_MODEL = 'gemini-2.5-flash'

PROMPT = prompt = f"""
You are a CUDA optimization expert. I will give you a kernel and profiler output.
Tell me the single biggest bottleneck and why. GPU specifications are RTX A4000 16gb ampere architechture.

CUDA KERNEL:
{cuda_code}

PROFILER OUTPUT:
{profile}

What is the ONE thing I should focus on?
"""

agent = Agent(
    name = 'cuda_agent',
    model = AGENT_MODEL,
    description="inspect cuda kernel",
    instruction=PROMPT
)

session_service = InMemorySessionService()

APP_NAME = 'CUDA'
USER_ID = 'user_1'
SESSION_ID = 'session_001'

async def init_session(app_name:str,user_id:str,session_id:str) -> InMemorySessionService:
    session = await session_service.create_session(
        app_name=app_name,
        user_id=user_id,
        session_id=session_id
    )
    print(f"Session created: App='{app_name}', User='{user_id}', Session='{session_id}'")
    return session

session = asyncio.run(init_session(APP_NAME,USER_ID,SESSION_ID))

runner = Runner(
    agent = agent,
    app_name=APP_NAME,
    session_service=session_service
)

print(f"Runner created for agent '{runner.agent.name}'.")

async def chat(query: str, runner, user_id, session_id):
    print(f"\n >>>user query {query}")

    content = types.Content(role='user', parts=[types.Part.from_text(text=query)])

    final_response_text = "Agent did not produce a final response."

    async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
        # print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate: 
                final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
                break

    print(f"<<< Agent Response: {final_response_text}")

if __name__ == "__main__":
    try:
        asyncio.run(chat("What is the biggest bottleneck in my kernel?", runner, USER_ID, SESSION_ID))
    except Exception as e:
        print(f"An error occurred: {e}")