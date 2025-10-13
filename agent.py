from langchain.chat_models import init_chat_model
from prompts import system_prompt_template, code_generation_prompt_template_geninfo, code_generation_prompt_template_modification, code_generation_prompt_template_plotting
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from base_models import Code
from langchain_core.prompts import PromptTemplate
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

        # define tools for agent to use
        self.tools = [
            Tool(
                name="CodeGeneration",
                func=self.__get_code, 
                description=(
                    "Tool for generating python code for only one task. "
                    "Returns a string containing python code"
                ), 
            ), 
            Tool(
                name="CodeExecutionPlotting", 
                func=self.__execute_code_plotting, 
                description=(
                    "Tool for executing code involving plotting. " 
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
                    "Tool for executing code that involves asking a question about the dataframe (e.g. who is the oldest person or what is the standard deviation of column x). "
                    "Returns a string representing the output"
                ), 
            )
        ]

        # define the agent_executor
        self.agent_executor = create_react_agent(self.model, self.tools)

    
    def __is_code_valid(self, code: Code, 
                        requires_visualization: bool, 
                        requires_modification: bool):
        # check if the llm added a field for the code as requried by the base model
        if 'code' not in code.keys():
            print('no code key avaiable')
            return False
        if requires_visualization:
            # check if plt_figure is at the last line
            code_string = code['code']
            return "plt_figure" in code_string.strip().split("\n")[-1]
        elif requires_modification == False:
            # check if there is a print statement at the end
            code_string = code['code']
            return "print" in code_string.strip().split("\n")[-1]
        return True

    def __code_generation(self, task: str, 
                          prompt_template: PromptTemplate,  
                          requires_visualization: bool, 
                          requires_modification: bool):
            code_generation_prompt = prompt_template.invoke({
                "columns": str(self.df.columns.tolist()).replace('[', '').replace(']', '').replace("'", ''), 
                "task": task
            })
            code_model = self.model.with_structured_output(Code)
            for _ in range(5):    
                code = code_model.invoke(code_generation_prompt)
                if self.__is_code_valid(code, requires_visualization, requires_modification):
                    return code['code']
                print('Code failed... Trying to generate again')
            return 'print("Sorry... Could not generate code after five tries")'

    # tool_1
    def __get_code(self, task: str, 
                   requires_visualization: bool, 
                   requires_modification: bool):
        if requires_visualization == True:
            code = self.__code_generation(
                task=task,
                prompt_template=code_generation_prompt_template_plotting, 
                requires_visualization=requires_visualization, 
                requires_modification=requires_modification)
            return code
        elif requires_modification == True:
            code = self.__code_generation(
                task=task,
                prompt_template=code_generation_prompt_template_modification, 
                requires_visualization=requires_visualization, 
                requires_modification=requires_modification)
            return code
        else:
            code = self.__code_generation(
                task=task,
                prompt_template=code_generation_prompt_template_geninfo, 
                requires_visualization=requires_visualization, 
                requires_modification=requires_modification)
            return code

    # tool_2
    def __execute_code_plotting(self, code: str) -> np.ndarray:
        code_execution_tool = PythonAstREPLTool(locals={"df": self.df})
        code_execution_tool.invoke(code) # execute the code
        return base64encoding(code_execution_tool.locals['plt_figure']) # get a base64 encoded plot

    # tool_3
    def __execute_code_modifying(self, code: str) -> pd.DataFrame:
        code_execution_tool = PythonAstREPLTool(locals={"df": self.df})
        code_execution_tool.invoke(code) # execute the code
        self.df = code_execution_tool.locals['df'] 
        return csv_encoding(self.df) # get the new encoded df
    
    # tool_4
    def __execute_code_geninfo(self, code: str) -> str:
        code_execution_tool = PythonAstREPLTool(locals={"df": self.df})
        output = code_execution_tool.invoke(code) # execute the code
        return output # get output (usually in a print statemnet)     


    # method to expose to MCP server
    def get_response(self, prompt):
        self.messages.append(HumanMessage(content=prompt))

        # invoke the agent executor
        response = self.agent_executor.invoke({"messages": self.messages})

        # update the messages list
        self.messages = response['messages']

        # return self.messages[-1].content
        return response