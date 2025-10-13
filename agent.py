from langchain.chat_models import init_chat_model
from prompts import task_prompt_template, code_generation_prompt_template_geninfo, code_generation_prompt_template_modification, code_generation_prompt_template_plotting, output_prompt_template, insights_prompt_template
from base_models import Tasks, Code, Task
from langchain_core.prompts import PromptTemplate
from langchain_experimental.tools import PythonAstREPLTool
import numpy as np
import pandas as pd
from utils import base64encoding, csv_encoding
from dotenv import load_dotenv

class Agent():
    def __init__(self, df):
        load_dotenv()
        self.model = init_chat_model("gpt-4o-mini", model_provider="openai")
        self.task_model = self.model.with_structured_output(Tasks)
        self.messages = [] # init messages for shared short-term memory
        self.df = df
    
    def __divide_tasks(self, prompt: str) -> Task:
        task_prompt = task_prompt_template.invoke({
            "columns": str(self.df.columns.tolist()).replace('[', '').replace(']', '').replace("'", ''), 
            "user_query": prompt, 
            "prev_context": str(self.messages[-3:]).replace('[', '').replace(']', '')
        })

        tasks = self.task_model.invoke(task_prompt) # get the required tasks

        # make sure the insights (if any) are in the end to allow code execution first
        tasks.tasks = sorted(tasks.tasks, key=lambda x: x.requires_code_execution, reverse=True)

        return tasks
    
    def __is_code_valid(self, code: Code, task: Task):
        # check if the llm added a field for the code as requried by the base model
        if 'code' not in code.keys():
            print('no code key avaiable')
            return False
        if task.requires_visualization:
            # check if plt_figure is at the last line
            code_string = code['code']
            return "plt_figure" in code_string.strip().split("\n")[-1]
        elif task.requires_modification == False:
            # check if there is a print statement at the end
            code_string = code['code']
            return "print" in code_string.strip().split("\n")[-1]
        return True

    def __code_generation(self, prompt_template: PromptTemplate, task: Task):
            code_generation_prompt = prompt_template.invoke({
                "columns": str(self.df.columns.tolist()).replace('[', '').replace(']', '').replace("'", ''), 
                "task": task.task_string
            })
            code_model = self.model.with_structured_output(Code)
            for _ in range(5):    
                code = code_model.invoke(code_generation_prompt)
                if self.__is_code_valid(code, task):
                    return code['code']
                print('Code failed... Trying to generate again')
            return 'print("Sorry... Could not generate code after five tries")'

    def __get_code(self, task: Task):
        if task.requires_visualization == True:
            code = self.__code_generation(
                prompt_template=code_generation_prompt_template_plotting, 
                task=task)
            return code
        elif task.requires_modification == True:
            code = self.__code_generation(
                prompt_template=code_generation_prompt_template_modification, 
                task=task)
            return code
        else:
            code = self.__code_generation(
                prompt_template=code_generation_prompt_template_geninfo, 
                task=task)
            return code

    def __execute_code_plotting(self, code) -> np.ndarray:
        code_execution_tool = PythonAstREPLTool(locals={"df": self.df})
        code_execution_tool.invoke(code) # execute the code
        return code_execution_tool.locals['plt_figure'] # get the numpy array representing the plot

    def __execute_code_modifying(self, code) -> pd.DataFrame:
        code_execution_tool = PythonAstREPLTool(locals={"df": self.df})
        code_execution_tool.invoke(code) # execute the code
        return code_execution_tool.locals['df'] # get the new df
    
    def __execute_code_geninfo(self, code) -> str:
        code_execution_tool = PythonAstREPLTool(locals={"df": self.df})
        output = code_execution_tool.invoke(code) # execute the code
        return output # get output (usually in a print statemnet)     

    def __get_friendly_output(self, prompt, geninfo_tasks, requires_visualization, requires_modification):
        friendly_output_prompt = output_prompt_template.invoke(
            {
                "prompt": prompt, 
                "answers": "".join(geninfo_tasks), 
                "requires_visualization": requires_visualization, 
                "requires_modification": requires_modification, 
                "context": str(self.messages[-3:]).replace('[', '').replace(']', '')
            }
        )
        friendly_output = self.model.invoke(friendly_output_prompt)
        return friendly_output.content

    def __get_insights(self, prompt):
        insights_prompt = insights_prompt_template.invoke({
            "prompt": prompt,
            "context": str(self.messages[-3:]).replace('[', '').replace(']', '')
        })
        insights = self.model.invoke(insights_prompt)
        return insights.content

    def get_response(self, prompt):

        # first phase: divide into task (assume all tasks are independent)
        tasks = self.__divide_tasks(prompt)

        # second phase: loop through all tasks
        #    - generate code for each task
        #    - get output for each task

        outputs = {
            "text": "", 
            "plots": [], 
            "df": ""
        }
        geninfo_tasks = []
        requires_visualization = sum(map(lambda x: x.requires_visualization, tasks.tasks)) >= 1
        requires_modification = sum(map(lambda x: x.requires_modification, tasks.tasks)) >= 1
        requires_code_execution = sum(map(lambda x: x.requires_code_execution, tasks.tasks)) >= 1
        insights = ""
        for task in tasks.tasks:
            # get code for task
            if not task.requires_code_execution:
                # give insights
                insights = self.__get_insights(prompt=prompt)
            else:   
                code = self.__get_code(task) 
                if task.requires_visualization:
                    plot = self.__execute_code_plotting(code)
                    base64plot = base64encoding(plot)
                    outputs['plots'].append(base64plot)
                elif task.requires_modification:
                    self.df = self.__execute_code_modifying(code)
                    encoded_df = csv_encoding(self.df)
                    outputs['df'] = encoded_df
                else:
                    output = self.__execute_code_geninfo(code)
                    task_output = f"Task: {task.task_string}\nOutput: {output}\n"
                    geninfo_tasks.append(task_output)
        
        # third phase: Generate a friendly output for text
        code_response = ""
        if requires_code_execution:
            code_response = self.__get_friendly_output(
                prompt=prompt, 
                geninfo_tasks=geninfo_tasks, 
                requires_visualization=requires_visualization, 
                requires_modification=requires_modification
            )
        
        # add the previous messages inclduing the human question and the AI response
        self.messages.append(
            {
                "Human": prompt, 
                "Assistant": code_response + insights
            }
        )

        outputs['text'] = code_response + insights

        return outputs