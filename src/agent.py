from langchain.chat_models import init_chat_model
from prompts import system_prompt_template
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_experimental.tools import PythonAstREPLTool
from langchain.agents import Tool
import numpy as np
import pandas as pd
from utils import base64encoding, csv_encoding
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent


class AgentReAct():
    def __init__(self, df):
        load_dotenv()
        self.model = init_chat_model("gpt-4o-mini", model_provider="openai")
        self.df = df
        self.messages = [SystemMessage(content=system_prompt_template.invoke({
            "columns":str(self.df.columns.tolist()).replace('[', '').replace(']', '').replace("'", '')
        }).text)] # init messages
        
        self.plots = []

        # define tools for agent to use
        self.tools = [
            Tool(
                name="CodeExecutionPlotting", 
                func=self.__execute_code_plotting, 
                description=(
                    "Tool for executing code involving plotting. It expects code generated from the CodeGeneration tool. " 
                    "Returns a base64 encoded image representing the plot"
                ), 
            ), 
            Tool(
                name="CodeExecutionModification", 
                func=self.__execute_code_modifying, 
                description=(
                   "Tool for executing code involving dataframe 'df' modification. "
                    "Returns a base64 encoded file representing a new csv file and internally modifies the dataframe"
                ),
            ), 
            Tool(
                name="CodeExecutionGenInfo", 
                func=self.__execute_code_geninfo, 
                description=(
                    "Tool for executing code that involves asking a question about the dataframe (e.g. who is the oldest person or what is the standard deviation of column x). It expects code generated from the CodeGeneration tool. "
                    "Returns a string representing the output"
                ), 
            )
        ]

        # define the agent_executor
        self.agent_executor = create_react_agent(self.model, self.tools)
    
    # tool_1
    def __execute_code_plotting(self, code: str):
        self.plots = []
        code_execution_tool = PythonAstREPLTool(locals={"df": self.df, "pd": pd, "np": np})
        code_execution_tool.invoke(code) # execute the code
        self.plots.append(base64encoding(code_execution_tool.locals['plt_figure'])) # add the plot
        return "generated plot successfully"

    # tool_3
    def __execute_code_modifying(self, code: str):
        code_execution_tool = PythonAstREPLTool(locals={"df": self.df, "pd": pd, "np": np})
        code_execution_tool.invoke(code) # execute the code
        self.df = code_execution_tool.locals['df'] 
        return "modified df successfully" # get the new encoded df
    
    # tool_4
    def __execute_code_geninfo(self, code: str) -> str:
        code_execution_tool = PythonAstREPLTool(locals={"df": self.df, "pd": pd, "np": np})
        output = code_execution_tool.invoke(code) # execute the code
        return output # get output (usually in a print statemnet)     

    def get_response(self, prompt):
        self.messages.append(HumanMessage(content=prompt))
        self.plots = []
        # invoke the agent executor
        response = self.agent_executor.invoke({"messages": self.messages})

        # update the messages list
        self.messages = response['messages']

        # return output
        return {
            "text": self.messages[-1].content, 
            "plots": self.plots, 
            "df": self.df.to_csv(index=False)
        }