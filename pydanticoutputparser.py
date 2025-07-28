from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    
    name: str = Field(description='name of the person')

    age: int = Field(gt= 18, description='age of the person')

    city: str = Field(description='name of the city the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name , age , and city of the fictional {place} person \n {format_instruction}",
    input_variables=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'place': 'indian'})

print(result)