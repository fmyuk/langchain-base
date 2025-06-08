from typing import Annotated, Literal, TypedDict

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from PIL import Image
import matplotlib.pyplot as plt
import io
import config

@tool
def search(query: str):
  """Call to surf the web."""
  return ["It's sunny in San Francisco."]

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

app = workflow.compile()

png_data = app.get_graph().draw_mermaid_png()
image = io.BytesIO(png_data)

img = Image.open(image)

plt.imshow(img)
plt.axis("off")
plt.show()