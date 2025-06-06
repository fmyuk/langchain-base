from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from scipy.spatial.distance import cosine
import config

embeddings_model = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY, model="text-embedding-3-small")

documents = [
  Document(page_content="今日は天気がいいですね。公園に行きますか？"),
  Document(page_content="AI技術は近年急速に発展しています。"),
  Document(page_content="コーヒーはとても人気があります。"),
  Document(page_content="Pythonはとても便利だ。")
]

db = Chroma.from_documents(documents=documents, embedding=embeddings_model)

query = "AIの発展について教えて"

query_embedding = embeddings_model.embed_query(query)

result = db.similarity_search_by_vector(query_embedding, k=1)

for doc in result:
  doc_embedding = embeddings_model.embed_query(doc.page_content)
  similarity = 1 - cosine(query_embedding, doc_embedding)
  print(f"Document: {doc.page_content}, Similarity: {similarity}")