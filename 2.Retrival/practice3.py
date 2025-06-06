from langchain_openai import OpenAIEmbeddings
from langchain_community.utils.math import cosine_similarity
import config

embeddings_model = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY, model="text-embedding-3-small")

embeddings1 = embeddings_model.embed_query("AIはどのように機能しますか？")
embeddings2 = embeddings_model.embed_query("人工知能の仕組みを教えてください。")
embeddings3 = embeddings_model.embed_query("天気はどうですか？")

similarity_1_2 = cosine_similarity([embeddings1], [embeddings2])
similarity_1_3 = cosine_similarity([embeddings1], [embeddings3])

print(f"similarity_1_2: {similarity_1_2[0][0]}")
print(f"similarity_1_3: {similarity_1_3[0][0]}")