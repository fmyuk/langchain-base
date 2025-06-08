from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from PIL import Image
import matplotlib.pyplot as plt
import io
import config
from serpapi import GoogleSearch

@tool
def search(query: str):
  """Search the web using SerpAPI."""
  params = {
    "q": query,
    "hl": "en",
    "gl": "us",
    "api_key": config.SERP_API_KEY
  }
  
  search = GoogleSearch(params)
  result = search.get_dict()
  
  results_list = result.get("organic_results", [])
  search_results = [
    f"{res['title']}: {res['snippet']} - {res['link']}" for res in results_list[:3]
  ]
  return search_results if search_results else ["No results found."]

tools = [search]

tool_node = ToolNode(tools)

model = ChatOpenAI(api_key=config.OPENAI_API_KEY, model_name="gpt-4o-mini").bind_tools(tools)

def should_continue(state: MessagesState) -> Literal["tools", END]:
  messages = state["messages"]
  last_message = messages[-1]
  if last_message.tool_calls:
    return "tools"
  return END

def call_model(state: MessagesState):
  messages = state["messages"]
  response = model.invoke(messages)
  
  return {"messages": [response]}

workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
  "agent",
  should_continue
)

workflow.add_edge("tools", "agent")

checkpointer = MemorySaver()

app = workflow.compile(checkpointer=checkpointer)

thread = {"configurable": {"thread_id": 42}}
inputs = [HumanMessage(content="what is the weather in San Francisco?")]

for event in app.stream({"messages": inputs}, thread, stream_mode="values"):
  event["messages"][-1].pretty_print()
  