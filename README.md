# Exploratory Data Analysis Agent
An AI agent developed that can preform data exploration in a csv file. The agent has three main features that are:
* Plotting: Has the ability to plot graphs (e.g. distribuion of a column)
* Modification: Has the ability to preform operations that involve modifying the csv file such as adding/removing columns or data cleaning as per the user request. 
* Question-Answering: Has the ability to answer general questions about the dataset (e.g. what is the standard deviation of column A)

# 1. Agent Architecture
The agent follows the reAct paradigm for reasoning. To find more, please read this [paper](https://arxiv.org/abs/2210.03629) in which this technique was proposed. Three main tools were exposed to the agent to be used, in which all involve code execution for the three main features. For code to be executed, the REPL tool was used, specifically the langchain wrapper [Python REPL](https://python.langchain.com/docs/integrations/tools/python/). As for the LLM, `gpt-4o-mini` was used due to its moderate reasoning cababilities and low cost.

# 2. Usage
To use the application locally, clone the repo:
```
git clone https://github.com/Kareem404/eda-agent.git
```
Create a `.env`to store the api key (make sure you obtain an API key from OpenAI). The content of the file should be as follows:
```
OPENAI_API_KEY=sk-proj-...
```
Install the requirements:
```
pip install -r requirements.txt
```
Run the application locally:
```
python agent_run.py
````
Make sure you modify the `agent_run.py` file to include the path of your csv file.
```
print('Reading dataset...')
df = pd.read_csv('Titanic-Dataset.csv') # add your path here
```
Please note that since you will be running the application in the CLI, you will not be able to see plots if you request any. Additionally, a new csv file `new_df.csv` that saves the modified df automatically after each request.  
# 3. MCP Server
An MCP server will be created to host this agent and you will be able to use the agent in claude desktop later!
