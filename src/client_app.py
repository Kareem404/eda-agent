import gradio as gr
import pandas as pd
import json
import base64
from io import BytesIO
from PIL import Image
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from utils import csv_encoding
from prompts import system_prompt_template
from dotenv import load_dotenv
load_dotenv()

# ========== GLOBAL VARIABLES ==========
client = None
tools = None
csv_upload_tool = None
agent = None
messages = []
df_memory = None

# ========== ASYNC INITIALIZATION ==========
async def setup_mcp():
    """Initialize the MCP client, tools, and agent once."""
    global client, tools, csv_upload_tool, agent
    # connect to MCP server
    client = MultiServerMCPClient({
        "EDA Agent": {
            "transport": "streamable_http",
            "url": "http://localhost:8001/mcp"
        },
    })
    tools = await client.get_tools()
    csv_upload_tool = tools[0]
    # load model
    model = init_chat_model("gpt-4o-mini", model_provider="openai")
    # create agent
    agent = create_react_agent(model, tools[1:])
    print("✅ MCP setup complete.")
    return True

# ========== CSV UPLOAD ==========
async def handle_csv_upload(file):
    """Upload CSV to MCP and store DataFrame in memory."""
    global df_memory, csv_upload_tool, messages
    if file is None:
        return "❌ No file uploaded."
    df_memory = pd.read_csv(file.name)
    base64_df = csv_encoding(df_memory)
    # upload to MCP
    result = await csv_upload_tool.ainvoke({"base64_csv": base64_df})
    # Initialize system prompt
    sys_msg = SystemMessage(content=system_prompt_template.invoke({
        "columns": str(df_memory.columns.tolist()).replace('[', '').replace(']', '').replace("'", '')
    }).text)
    messages = [sys_msg]
    preview = df_memory.head().to_markdown()
    return f"✅ CSV uploaded successfully to MCP.\n\n{result}\n\n**Preview:**\n{preview}"

# ========== PLOT RETRIEVAL ==========
async def get_plots_from_mcp():
    """Retrieve plots from MCP resources and decode them."""
    global client
    try:
        resources = await client.get_resources(server_name="EDA Agent")
        if len(resources) > 1:
            plots_data = resources[1].data
            plots_json = json.loads(plots_data)
            plots_list = plots_json.get("plots", [])
            
            if plots_list:
                # Decode base64 images
                images = []
                for plot_b64 in plots_list:
                    img_data = base64.b64decode(plot_b64)
                    img = Image.open(BytesIO(img_data))
                    images.append(img)
                return images
        return []
    except Exception as e:
        print(f"Error retrieving plots: {e}")
        return []

async def get_csv_from_mcp():
    """Retrieve processed CSV from MCP resources."""
    global client
    try:
        resources = await client.get_resources(server_name="EDA Agent")
        if len(resources) > 0:
            csv_base64 = resources[0].data
            # Decode base64 to get CSV content
            csv_data = base64.b64decode(csv_base64)
            # Save to temporary file for download
            temp_file = "processed_data.csv"
            with open(temp_file, "wb") as f:
                f.write(csv_data)
            return temp_file
        return None
    except Exception as e:
        print(f"Error retrieving CSV: {e}")
        return None

# ========== CHAT HANDLER ==========
async def chat_with_mcp(message, history):
    """Handles user messages and gets responses from MCP."""
    global messages, agent, df_memory
    if df_memory is None:
        return "⚠️ Please upload a CSV first before chatting.", None
    
    # Add user message
    messages.append(HumanMessage(content=message))
    # Get agent response
    response = await agent.ainvoke({"messages": messages})
    messages = response["messages"]
    
    # Get the text response
    text_response = messages[-1].content
    
    # Check for plots
    plots = await get_plots_from_mcp()
    
    return text_response, plots if plots else None

# ========== BUILD GRADIO INTERFACE ==========
with gr.Blocks() as demo:
    gr.Markdown("# 🧠 LangChain MCP Chatbot")
    # Initialize MCP connection button
    init_btn = gr.Button("🔗 Connect to MCP Server")
    init_status = gr.Markdown("")
    
    async def init_mcp_connection():
        await setup_mcp()
        return "✅ Connected to MCP and initialized agent."
    
    init_btn.click(init_mcp_connection, outputs=init_status)
    
    with gr.Tab("📂 Upload CSV"):
        file_input = gr.File(label="Upload your CSV file")
        upload_output = gr.Markdown()
        download_btn =  gr.Button("⬇️ Download CSV")

        file_input.upload(handle_csv_upload, inputs=file_input, outputs=upload_output)


        async def download_csv():
            csv_file = await get_csv_from_mcp()
            return csv_file if csv_file else None

        download_btn.click(download_csv, inputs=None, outputs=download_btn)
    
    with gr.Tab("💬 Chat"):
        with gr.Row():
            with gr.Column():
                chatbot = gr.Chatbot(label="Conversation", height=500)
                msg_input = gr.Textbox(
                    label="Your Message",
                    placeholder="Ask questions about your data...",
                    lines=2
                )
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear")
            
            with gr.Column():
                plot_gallery = gr.Gallery(
                    label="Generated Plots",
                    show_label=True,
                    columns=1,
                    height=500,
                    object_fit="contain"
                )
        
        async def respond(message, chat_history):
            if not message.strip():
                return chat_history, "", None
            
            # Add user message to history
            chat_history.append((message, None))
            
            # Get response and plots
            bot_response, plots = await chat_with_mcp(message, chat_history)
            
            # Update chat history with bot response
            chat_history[-1] = (message, bot_response)
            
            return chat_history, "", plots
        
        submit_btn.click(
            respond,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input, plot_gallery]
        )
        
        msg_input.submit(
            respond,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input, plot_gallery]
        )
        
        clear_btn.click(
            lambda: ([], None),
            outputs=[chatbot, plot_gallery]
        )

demo.launch()