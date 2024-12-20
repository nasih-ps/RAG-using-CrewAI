from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from crewai_tools import BaseTool
from crewai import Agent, Task, Crew, Process, LLM
from typing import Any
from pydantic import Field
load_dotenv()

class PDFSearchSystem:
    def __init__(self, pdf_path):

        os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
        self.embedding = AzureOpenAIEmbeddings(
            model=os.environ["AZURE_OPENAI_EMBED_MODEL"],
            azure_endpoint=os.environ["AZURE_OPENAI_EMBED_API_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_EMBED_API_KEY"],
            openai_api_version=os.environ["AZURE_OPENAI_EMBED_VERSION"]
        )
        
        self.llm = LLM(
            model="azure/gpt-4o-001",
            temperature=0.7,
            base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
        
        self.vectorstore = self._process_pdf(pdf_path)
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
        )
        
    def _process_pdf(self, pdf_path):

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        vectorstore = FAISS.from_documents(chunks, self.embedding)
        print("Successfully created vector store from PDF")
        return vectorstore
    
    def search_info(self, query: str) -> str:
        try:
            docs = self.retriever.invoke(query)
            if not docs:
                return "No information found"
            return "\n".join(doc.page_content for doc in docs)
        except Exception as e:
            return f"Error during search: {str(e)}"

class PDFSearchTool(BaseTool):
    name: str = "PDF Search Tool"
    description: str = "Search information from the loaded PDF."
    pdf_search_system: Any = Field(default=None, exclude=True)
    
    def __init__(self, pdf_search_system: PDFSearchSystem, **data):
        super().__init__(**data)
        self.pdf_search_system = pdf_search_system
    
    def _run(self, query: str) -> str:
        if isinstance(query, dict):
            query = query.get('query', '') if isinstance(query.get('query'), str) else str(query)
        return self.pdf_search_system.search_info(query)

def setup_pdf_search_system(pdf_path: str):

    pdf_system = PDFSearchSystem(pdf_path)
    
    tools = [PDFSearchTool(pdf_search_system=pdf_system)]
    
    search_agent = Agent(
        role='PDF information search',
        goal='Find answer from the PDF. Reply "No information" if information cannot be found',
        backstory="""Search through the PDF content to find relevant information.""",
        llm=pdf_system.llm,
        verbose=True,
        allow_delegation=True,
        tools=tools
    )
    
    return search_agent

def search_pdf(agent, question: str):

    task = Task(
        description=question,
        expected_output="""relevant information from PDF""",
        agent=agent
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential
    )
    
    return crew.kickoff()

if __name__ == "__main__":

    pdf_path = "./random.pdf"
    
    agent = setup_pdf_search_system(pdf_path)
    
    question = "What is the main topic of the document?" 
    result = search_pdf(agent, question)
    print("######################")
    print(result)