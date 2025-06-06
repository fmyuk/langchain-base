from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os
import config

main_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(main_path)

loader = PyPDFLoader("LangChain株式会社IR資料.pdf")
documents = loader.load()

whole_document = "".join([page.page_content for page in documents])

llm = ChatOpenAI(api_key=config.OPENAI_API_KEY, model_name="gpt-4o-mini")

template = """
あなたはPDFドキュメントに基づいて質問に回答するアシスタントです。以下のドキュメントに基づいて質問に回答してください。

ドキュメント: {document}

質問: {question}

回答:
"""

prompt = PromptTemplate(input_variables=["document", "question"], template=template)

def chatbot(question):
  filled_prompt = prompt.format(document=whole_document, question=question)

  response = llm.invoke(filled_prompt)

  return response

question = "LangChain株式会社の最近の業績は？"
response = chatbot(question)

print(response)