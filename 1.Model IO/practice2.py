from langchain_openai import ChatOpenAI
import config

llm = ChatOpenAI(api_key=config.OPENAI_API_KEY, model="gpt-4o-mini", max_tokens=100, temperature=0.7)

messages = [
  (
    "system",
    "あなたは優秀なPythonエンジニアです。Pythonのコードを出力してください。"
  ),
  ("human", "PythonでHello Worldを出力するコードを書いてください。"),
]

ai_msg = llm.invoke(messages)
print(ai_msg)
