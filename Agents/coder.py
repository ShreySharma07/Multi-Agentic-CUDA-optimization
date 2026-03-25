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



AGENT_MODEL = 'gemini-3-flash-preview'

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
    tools=[]
)

session_service = InMemorySessionService()

APP_NAME = 'CUDA'
USER_ID = 'user_1'
SESSION_ID = 'session_001'

# async def init_session(app_name:str,user_id:str,session_id:str) -> InMemorySessionService:
#     session = await session_service.create_session(
#         app_name=app_name,
#         user_id=user_id,
#         session_id=session_id
#     )
#     print(f"Session created: App='{app_name}', User='{user_id}', Session='{session_id}'")
#     return session

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

session = asyncio.run(init_session(APP_NAME,USER_ID,SESSION_ID))

runner = Runner(
    agent = agent,
    app_name=APP_NAME,
    session_service=session_service
)

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
    print(f"\n >>> User Query: {query}")
    content = types.Content(role='user', parts=[types.Part.from_text(text=query)])
    
    max_retries = 3
    retry_delay = 10  
    
    for attempt in range(max_retries):
        try:
            full_response = "" # 1. Create a variable to hold the complete response
            
            async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, "text") and part.text:
                            # Keep printing so you can watch it live in the terminal
                            print(part.text, end="", flush=True) 
                            # 2. Add the text chunk to our variable
                            full_response += part.text 
            
            print("\n")
            return full_response # 3. Return the full string back to main.py!
            
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                print(f"\n⚠️ Rate limit hit (Attempt {attempt + 1}/{max_retries}).")
                print(f"Waiting {retry_delay}s before retrying...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  
            else:
                print(f"\n A non-quota error occurred: {e}")
                break
    
    return "" # Return empty string if all retries fail

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