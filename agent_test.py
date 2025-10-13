from agent import Agent
import pandas as pd

print('Reading dataset...')
df = pd.read_csv('Titanic-Dataset.csv')

print("Loading Agent...")
bot = Agent(df=df)

print('Agent Loaded!')
while True:
    user_input = input("Enter: ")
    if user_input == "quit":
        break
    response = bot.get_response(user_input)
    print(f"agent: {response['text']}")