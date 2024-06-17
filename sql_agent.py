from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain.agents import create_sql_agent
from langchain.agents import AgentType

llm = Ollama(model="llama3")

# for chunk in llm.stream("Introduce yourself"):
#     print(chunk, end="")


db = SQLDatabase.from_uri("sqlite:///chinook.db")

# print(db.get_usable_table_names())
# print(db.get_table_info(['albums']))
# print(db.table_info)


agent = create_sql_agent(llm=llm, db=db, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# response = agent.invoke({"user": "How many different artists are there in the database"})
for chunk in agent.stream("List the total sales per country. Which country's customers spent the most?"):
    print(chunk, end="")
