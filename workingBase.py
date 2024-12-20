from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
import os
from dotenv import load_dotenv
from crewai_tools import BaseTool
from crewai import Agent, Task, Crew, Process, LLM
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
embedding = AzureOpenAIEmbeddings(
    model = os.environ["AZURE_OPENAI_EMBED_MODEL"],
    azure_endpoint = os.environ["AZURE_OPENAI_EMBED_API_ENDPOINT"],
    api_key = os.environ["AZURE_OPENAI_EMBED_API_KEY"],
    openai_api_version = os.environ["AZURE_OPENAI_EMBED_VERSION"]
)

llm = LLM(
    model="azure/gpt-4o-001",
    temperature=0.7,
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

test_texts = ["harrison worked at kensho"]
try:  
    test_embedding = embedding.embed_documents(test_texts)
    print("Successfully generated test embedding")
    vectorstore = FAISS.from_texts(
        texts=test_texts,
        embedding=embedding
    )
    print("Successfully created vector store")
except Exception as e:
    print(f"Detailed error: {str(e)}")
    raise

retriever = vectorstore.as_retriever()

class WorkInfoSearchTool(BaseTool):
    name: str = "Work Info Search Tool"
    description: str = "Search work related information."
    
    def _run(self, query: str) -> str:
        # Handle both string and dict inputs
        if isinstance(query, dict):
            query = query.get('query', '') if isinstance(query.get('query'), str) else str(query)
        
        try:
            # Using the new invoke() method instead of get_relevant_documents()
            docs = retriever.invoke(query)
            if not docs:
                return "No information found"
            return "\n".join(doc.page_content for doc in docs)
        except Exception as e:
            return f"Error during search: {str(e)}"

tools = [WorkInfoSearchTool()]

websearch_agent = Agent(
    role='Work information search',
    goal='Find answer for work information. Reply "No information" if you cannot find information',
    backstory="""Find answer for work information relevant to the user.""",
    llm=llm,
    verbose=True,
    allow_delegation=True,
    tools=tools
)

task1 = Task(
    description="""where did harrison work?""",
    expected_output="""work information""",
    agent=websearch_agent
)

crew = Crew(
    agents=[websearch_agent],
    tasks=[task1],
    process=Process.sequential
)

result = crew.kickoff()
print("######################")
print(result)