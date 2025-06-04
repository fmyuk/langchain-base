from langchain_openai import ChatOpenAI
from langchain_core.propmts import ChatPromptTemplate
import config

llm = ChatOpenAI(api_key=config.OPENAI_API_KEY, model="gpt-4o-mini")

template = ChatPropmtTemplate([
    ("system", "あなたは優秀なPythonエンジニアです。Pythonのコードを出力してください。"),
    ("human", "{user_input}"),
])

prompt_value = template.invoke(
  {
    "user_input": "PythonでHello Worldを出力するコードを書いてください。"
  }
)

response = llm.invoke(prompt_value)

print(response)