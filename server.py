from fastmcp import FastMCP
import pandas as pd
import base64
from io import StringIO
from utils import csv_encoding, base64encoding
from langchain_experimental.tools import PythonAstREPLTool
import numpy as np

# instantiate the server
mcp = FastMCP("EDA Agent")

df = None
plots = []

@mcp.tool()
def upload_csv(base64_csv: str):
    """Tool for uploading a csv file and saving it in memory"""
    decoded_str = base64.b64decode(base64_csv).decode('utf-8')
    global df # signals to modify the global df
    df = pd.read_csv(StringIO(decoded_str))
    return {
        "Message": "CSV file read successfully"
    }

@mcp.resource("resource://csv_file")
def get_csv():
    """Provides the csv file as base64 encoded string"""
    if df is not None:
        return csv_encoding(df)
    return None

@mcp.tool()
def execute_code_geninfo(code: str):
    """
    Tool for executing code that involves asking a question about the dataframe (e.g. who is the oldest person or what is the standard deviation of column x). It expects code generated from the CodeGeneration tool.
    Returns a string representing the output
    """
    try:
        if df is not None:
            code_execution_tool = PythonAstREPLTool(locals={"df": df, "pd": pd, "np": np})
            output = code_execution_tool.invoke(code) # execute the code
            return {
                "output": output
            }
        else:
            return {
                "Error": "No csv file uploaded. Please upload csv file first"
            }
    except Exception as e:
        return {"Error": e}


@mcp.tool()
def execute_code_modifying(code: str):
    """
    Tool for executing code involving dataframe 'df' modification.
    Returns a base64 encoded file representing a new csv file and internally modifies the dataframe
    """
    try:
        global df
        if df is not None:
            code_execution_tool = PythonAstREPLTool(locals={"df": df, "pd": pd, "np": np})
            code_execution_tool.invoke(code) # execute the code
            df = code_execution_tool.locals['df']
            return {
                "Message": "df modified successfully"
            }
        else:
            return {
                "Error": "No csv file uploaded. Please upload csv file first"
            }
    except Exception as e:
        return {"Error": e}


@mcp.tool()
def execute_code_plotting(code: str):
    """
        "Tool for executing code that involves asking a question about the dataframe (e.g. who is the oldest person or what is the standard deviation of column x). It expects code generated from the CodeGeneration tool.
        Returns a string representing the output
    """
    try:
        if df is not None:
            global plots
            plots = []
            code_execution_tool = PythonAstREPLTool(locals={"df": df, "pd": pd, "np": np})
            code_execution_tool.invoke(code) # execute the code
            plots.append(base64encoding(code_execution_tool.locals['plt_figure'])) # add the plot
            return {
                "Message": "plots generated successfully"
            }
        else:
            return {
                "Message": "No csv file uploaded. Please upload csv file first"
            }
    except Exception as e:
        return {"Message": e}

@mcp.resource("resource://plots")
def get_plots():
    """
    provides a list of plots encoded as base64 strings
    """
    global plots
    return {
        "plots": plots
    }

if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=8001)
