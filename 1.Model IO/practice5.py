from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import config

llm = ChatOpenAI(api_key=config.OPENAI_API_KEY, model="gpt-4o-mini")

template = "あなたは優秀なPythonエンジニアです。次の質問に答えてください。\n\n質問: {question}\n\n回答:"
prompt = PromptTemplate(input_variables=["question"], template=template)

filled_prompt = prompt.format(question="PythonでHello Worldを出力するコードを書いてください。")
response = llm.invoke(filled_prompt)

parsed_response = StrOutputParser().invoke(response)

print(parsed_response)