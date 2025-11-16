# https://github.com/langchain-ai/langchain-mcp-adapters

from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
import pandas as pd
from utils import csv_encoding
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from prompts import system_prompt_template
import asyncio
from dotenv import load_dotenv
load_dotenv()

# connect to server
client = MultiServerMCPClient(
    {
        "EDA Agent": {
            "transport": "streamable_http",
            "url": "http://localhost:8001/mcp"
        },
    }
)

# get MCP tools
async def get_tools():
    tools = await client.get_tools()
    return tools

async def upload_csv(base64_df: str, csv_upload_tool):
    result = await csv_upload_tool.ainvoke({"base64_csv": base64_df})
    return result

async def main():
    tools = await get_tools()
    print('got tools successfully')
    csv_upload_tool = tools[0]

    # load model
    model = init_chat_model("gpt-4o-mini", model_provider="openai")

    # upload df (this will be modified later to allow the user to upload whatever they want)
    df = pd.read_csv('Titanic-Dataset.csv')
    base64_df = csv_encoding(df)

    message = await upload_csv(base64_df=base64_df, csv_upload_tool=csv_upload_tool)
    print(message)

    messages = [SystemMessage(content=system_prompt_template.invoke({
                "columns":str(df.columns.tolist()).replace('[', '').replace(']', '').replace("'", '')
            }).text)] # init messages



    # create agent
    agent = create_react_agent(model, tools[1:])
    print('loaded agent successfully')

    # agentic loop
    while True: 
        prompt = input("Enter: ")
        messages.append(HumanMessage(content=prompt))
        response = await agent.ainvoke({"messages": messages})
        messages = response['messages']
        print(f"Agent: {messages[-1].content}")

if __name__ == "__main__":
    asyncio.run(main())