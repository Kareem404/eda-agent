from agent import AgentReAct
import pandas as pd

print('Reading dataset...')
df = pd.read_csv('Titanic-Dataset.csv')

print("Loading Agent...")
bot = AgentReAct(df=df)

print('Agent Loaded!')
while True:
    user_input = input("Enter: ")
    if user_input == "quit":
        break
    response = bot.get_response(user_input)
    print(f"agent: {response['text']}")
    print(f"plots: {len(response['plots'])}")
    # save dataframe locally
    new_df = response['df']
    new_df.to_csv('new-df.csv', index=False)