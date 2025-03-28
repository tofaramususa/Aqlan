# from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


llm = ChatGroq(temperature=0, model_name="llama3-8b-8192")
llm_json_mode = llm.with_structured_output(method="json_mode")