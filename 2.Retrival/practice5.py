from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import os
import config

main_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(main_path)

loader = PyPDFLoader("LangChain株式会社IR資料.pdf")
documents = loader.load()

embeddings_model = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY, model="text-embedding-3-small")

db = Chroma.from_documents(documents=documents, embedding=embeddings_model)

llm = ChatOpenAI(api_key=config.OPENAI_API_KEY, model_name="gpt-4o-mini")

template = """
あなたはPDFドキュメントに基づいて質問に回答するアシスタントです。以下のドキュメントに基づいて質問に回答してください。

ドキュメント: {document_snippet}

質問: {question}

回答:
"""

prompt = PromptTemplate(input_variables=["document_snippet", "question"], template=template)

def chatbot(question):
  question_embedding = embeddings_model.embed_query(question)
  document_snippet = db.similarity_search_by_vector(question_embedding, k=3)
  print(f"document_snippet: {document_snippet}")
  filled_prompt = prompt.format(document_snippet=document_snippet, question=question)
  response = llm.invoke(filled_prompt)
  return response

question = "LangChain株式会社の最近の業績は？"
response = chatbot(question)

print(response)