from pydantic import BaseModel, Field


code_execution_bool_description = """
boolean that checks whether the taks requires code generation or not. This could be a task that requires analysis of the previous outputs or general questions that do not require code execution. Any analysis question of values previously calculated does not require execution
"""

#### Not Needed ####
class Task(BaseModel):
    """Information about a task"""
    task_string: str = Field(default=None, description="Contains the extracted individual task string")
    requires_visualization: bool = Field(default=None, description="boolean that check whether we need to plot a figure or not")
    requires_modification: bool = Field(default=None, description="boolean that checks whether we need to modify the df object or not")
    requires_code_execution: bool = Field(default=None, description=code_execution_bool_description)

class Tasks(BaseModel):
    """Extracted tasks"""
    tasks: list[Task]
#### Not Needed ####

class Code:
    code: str = Field("The code for the required task")

class get_code(BaseModel):
    task: str
    requires_visualization: bool 
    requires_modification: bool

class execute_code(BaseModel):
    code: str