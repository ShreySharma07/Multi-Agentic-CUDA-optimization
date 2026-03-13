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
from tools.getfiles import getFiles
from tools.readfiles import readFile
from tools.compile_cuda import compile_cuda
from tools.ncu_profile import run_ncu_profile
from tools.parse_profile import parse_ncu_profile


import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.ERROR)

DIRECTORY_PATH = Path("kernelpractise")
# CUDA_PATH = "kernelpractise/sigmoid_kernel.cu"

# try:
#     if os.path.exists(CUDA_PATH):
#         with open(CUDA_PATH, 'r') as f:
#             cuda_code = f.read()
# except Exception as e:
#     print(e)

# try:
#     if os.path.exists(PROFILE_PATH):
#         with open(PROFILE_PATH, 'r') as f:
#             profile = f.read()
# except Exception as e:
#     print(e)

# os.environ['GOOGLE_API_KEY'] = 'GOOGLE_API'



AGENT_MODEL = 'gemini-2.5-flash'

PROMPT = f"""
You are a CUDA optimization expert.

GPU: RTX A4000 16GB (Ampere architecture)

CUDA files are located in:
{DIRECTORY_PATH}

Workflow:

1. Use getFiles to list CUDA files.
2. Ask the user which file to analyze.
3. Use readFile to read the selected CUDA file.
4. Identify the biggest performance bottlenecks.
5. Rewrite the CUDA kernel with optimizations.

IMPORTANT RULES:

- Always output the FULL optimized CUDA kernel.
- The optimized kernel must be complete and compilable.
- Include the entire kernel code, not just a snippet.
- Wrap the final optimized kernel inside a ```cpp code block.
- Do not stop after explaining optimizations.

Final output format:

1. Short explanation of bottlenecks
2. Fully optimized CUDA kernel
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

        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    print(part.text, end="", flush=True)

    if event.is_final_response():
        print()

    print(f"<<< Agent Response: {final_response_text}")

if __name__ == "__main__":
    try:
        while True:
            query = input("\nuser query: ")

            if query in ['stop', 'exit']:
                break
            asyncio.run(chat(query, runner, USER_ID, SESSION_ID))
    except Exception as e:
        print(f"An error occurred: {e}")