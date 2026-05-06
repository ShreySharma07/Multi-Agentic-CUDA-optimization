from pydantic import BaseModel, field_validator
import pandas as pd
import torch
import torch.nn as nn
import ast
import time
import os
import subprocess
from pathlib import Path
import google
import asyncio
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.errors import already_exists_error
from tools.getfiles import getFiles
from tools.readfiles import readFile


import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.ERROR)

DIRECTORY_PATH = Path("kernels")

# AGENT_MODEL = "gemini-3.1-flash-lite-preview"
# AGENT_MODEL = "gemini-3-flash-preview"
# AGENT_MODEL = "claude-sonnet-4-6"
AGENT_MODEL = "gemini-2.5-flash"

PROMPT = f"""
You are a CUDA optimization expert for RTX A4000 (sm_86).

CUDA files are located in:
{DIRECTORY_PATH}

IMPORTANT RULES:
- Always output the FULL optimized CUDA kernel.
- The optimized kernel must be complete and compilable.

STRICT OUTPUT RULES — MUST FOLLOW:
- Output ONLY valid CUDA C++ code.
- DO NOT include:
  - explanations
  - comments outside code
  - markdown or code blocks (no ```cpp)
  - headings like "1. Bottlenecks"
- First line MUST start directly with: #include
- If you violate this, the system will reject your output.

VALIDATION RULE (VERY IMPORTANT):
- You must include a main() function that validates the GPU output against a CPU reference.
- Use tolerance-based comparison: abs(gpu - cpu) <= 1e-3 * max(1.0, abs(gpu), abs(cpu))
- At the very end of main(), you MUST print exactly two lines in this exact format:
  1. If math is correct: printf("SUCCESS\\n");
     If math is wrong: printf("FAILURE\\n");
  2. printf("GPU Time: %f\\n", milliseconds);
- DO NOT print the word "milliseconds" in the output string, just the number.
"""

agent = Agent(
    name = 'cuda_agent',
    model = AGENT_MODEL,
    description="inspect cuda kernel",
    instruction=PROMPT,
    tools=[getFiles, readFile]
)

session_service = InMemorySessionService()

APP_NAME = 'CUDA'
USER_ID = 'user_1'
SESSION_ID = 'session_001'

async def init_session(app_name: str, user_id: str, session_id: str):
    try:
        session = await session_service.create_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id
        )
        print(f"Session created: {session_id}")
    except already_exists_error:
        # If it exists, we just fetch it
        session = await session_service.get_session(
            app_name=app_name,
            user_id=user_id,
            session_id=session_id
        )
        print(f"Using existing session: {session_id}")
    return session

# session = asyncio.run(init_session(APP_NAME,USER_ID,SESSION_ID))

runner = Runner(
    agent = agent,
    app_name=APP_NAME,
    session_service=session_service
)

_session_initialized = False

async def ensure_session():
    global _session_initialized
    if not _session_initialized:
        await init_session(APP_NAME, USER_ID, SESSION_ID)
        _session_initialized = True

print(f"Runner created for agent '{runner.agent.name}'.")

# async def chat(query: str, runner, user_id, session_id):
#     print(f"\n >>>user query {query}")

#     content = types.Content(role='user', parts=[types.Part.from_text(text=query)])

#     final_response_text = "Agent did not produce a final response."

#     async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
#         # print(f"  [Event] Author: {event.author}, Type: {type(event).__name__}, Final: {event.is_final_response()}, Content: {event.content}")

#         if event.content and event.content.parts:
#             for part in event.content.parts:
#                 if hasattr(part, "text") and part.text:
#                     print(part.text, end="", flush=True)

#     if event.is_final_response():
#         print()

#     print(f"<<< Agent Response: {final_response_text}")

async def chat(query: str, runner, user_id, session_id):
    await ensure_session()
    content = types.Content(role='user', parts=[types.Part.from_text(text=query)])
    
    final_text = ""
    
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=content
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, 'text') and part.text and part.text.strip():
                    final_text += part.text  # += to concatenate both final parts
    
    return final_text if final_text.strip() else "Agent produced no output."

import asyncio

# async def safe_chat(prompt, runner, user_id, session_id):
#     max_retries = 5
    
#     for attempt in range(max_retries):
#         try:
#             return await chat(prompt, runner, user_id, session_id)
        
#         except Exception as e:
#             error_str = str(e)
            
#             if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
#                 wait_time = 50  # or parse from error
                
#                 print(f"[RATE LIMIT] Waiting {wait_time}s before retry...")
#                 await asyncio.sleep(wait_time)
#             else:
#                 raise e
    
#     return "Agent failed due to repeated rate limits."

import asyncio
import random

async def safe_chat(prompt, runner, user_id, session_id, max_retries=5):
    for attempt in range(max_retries):
        try:
            return await chat(prompt, runner, user_id, session_id)
        except Exception as e:
            err = str(e)
            if "ConnectError" in err or "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = 30 * (attempt + 1)  # 30s, 60s, 90s...
                print(f"  network/rate error — waiting {wait}s (attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(wait)
            else:
                raise e
    return ""

async def main():
    await init_session(APP_NAME, USER_ID, SESSION_ID)
    
    print(f"Runner created for agent '{runner.agent.name}'.")

    while True:
        try:
            query = input("\nuser query: ")

            if query.lower() in ['stop', 'exit']:
                break

            await chat(query, runner, USER_ID, SESSION_ID)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"An Unexptected Error Occured: {e}")

if __name__ == "__main__":
    asyncio.run(main())