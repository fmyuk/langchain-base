from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import config

model = ChatOpenAI(api_key=config.OPENAI_API_KEY, model="gpt-4o-mini")

class TransLateWords(BaseModel):
  english: str = Field(description="英語")
  french: str = Field(description="フランス語")
  chinese: str = Field(description="中国語")

query = "ありがとう"

parser = JsonOutputParser(pydantic_object=TransLateWords)

prompt = PromptTemplate(
  template="指定した言語に翻訳してください。\n{format_instructions}\n{query}\n",
  input_variables=["query"],
  partial_variables={
    "format_instructions": parser.get_format_instructions()
  }
)

chain = prompt | model | parser
# query_prompt = prompt.invoke({"query": query})

result = chain.invoke({"query": query})
# output = model.invoke(query_prompt)
# result = parser.invoke(output)

print(result)