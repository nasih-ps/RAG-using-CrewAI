from langchain_community.vectorstores import FAISS
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from crewai_tools import BaseTool
from crewai import Agent, Task, Crew, Process, LLM
from typing import Any
from pydantic import Field
import pandas as pd
load_dotenv()

class CSVSearchSystem:
    def __init__(self, csv_path):
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
        
        self.vectorstore = self._process_csv(csv_path)
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 3}  # Retrieve top 3 most relevant chunks
        )
        
    def _process_csv(self, csv_path):

        df = pd.read_csv(csv_path)
        
        documents = []
        for idx, row in df.iterrows():
            # Combine all columns into a single text string
            content = " ".join([f"{col}: {str(val)}" for col, val in row.items()])
            # Create a document with the row content
            documents.append({
                "page_content": content,
                "metadata": {"row": idx}
            })
        
        # Split documents if they're too long
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.create_documents(
            texts=[doc["page_content"] for doc in documents],
            metadatas=[doc["metadata"] for doc in documents]
        )
        
        vectorstore = FAISS.from_documents(chunks, self.embedding)
        print("Successfully created vector store from CSV")
        return vectorstore
    
    def search_info(self, query: str) -> str:
        try:
            docs = self.retriever.invoke(query)
            if not docs:
                return "No information found"
            return "\n".join(doc.page_content for doc in docs)
        except Exception as e:
            return f"Error during search: {str(e)}"

class CSVSearchTool(BaseTool):
    name: str = "CSV Search Tool"
    description: str = "Search information from the loaded CSV file."
    csv_search_system: Any = Field(default=None, exclude=True)
    
    def __init__(self, csv_search_system: CSVSearchSystem, **data):
        super().__init__(**data)
        self.csv_search_system = csv_search_system
    
    def _run(self, query: str) -> str:
        if isinstance(query, dict):
            query = query.get('query', '') if isinstance(query.get('query'), str) else str(query)
        return self.csv_search_system.search_info(query)

def setup_csv_search_system(csv_path: str):
    csv_system = CSVSearchSystem(csv_path)
    
    tools = [CSVSearchTool(csv_search_system=csv_system)]
    
    search_agent = Agent(
        role='CSV information search',
        goal='Find answer from the CSV file. Reply "No information" if information cannot be found',
        backstory="""Search through the CSV content to find relevant information.""",
        llm=csv_system.llm,
        verbose=True,
        allow_delegation=True,
        tools=tools
    )
    
    return search_agent

def search_csv(agent, question: str):
    task = Task(
        description=question,
        expected_output="""relevant information from CSV""",
        agent=agent
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential
    )
    
    return crew.kickoff()

if __name__ == "__main__":

    csv_path = "./variance-template.csv"
    
    agent = setup_csv_search_system(csv_path)
    
    question = "What are the main trends in the data?" 
    result = search_csv(agent, question)
    print("######################")
    print(result)