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

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.core.config import settings
from app.utils.logger import logger
from io import BytesIO
import PyPDF2
from langchain.docstore.document import Document

load_dotenv()

pdf_router = APIRouter()

# Global variable to store PDF content
global_pdf_content = ""

class PDFSearchSystem:
    def __init__(self, pdf_content: str):
        os.environ["OPENAI_API_KEY"] = settings.AZURE_OPENAI_API_KEY
        self.embedding = AzureOpenAIEmbeddings(
            model=settings.AZURE_OPENAI_EMBED_MODEL,
            azure_endpoint=settings.AZURE_OPENAI_EMBED_API_ENDPOINT,
            api_key=settings.AZURE_OPENAI_EMBED_API_KEY,
            openai_api_version=settings.AZURE_OPENAI_EMBED_VERSION
        )
        
        self.llm = LLM(
            model="azure/gpt-4o-001",
            temperature=0.7,
            base_url=settings.AZURE_OPENAI_ENDPOINT,
            api_key=settings.AZURE_OPENAI_API_KEY
        )
        
        self.vectorstore = self._process_pdf_content(pdf_content)
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        
    def _process_pdf_content(self, pdf_content: str):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(pdf_content)
        documents = [Document(page_content=chunk, metadata={}) for chunk in chunks]
        
        vectorstore = FAISS.from_documents(documents, self.embedding)
        logger.info("Successfully created vector store from PDF content")
        return vectorstore
    
    def search_info(self, query: str) -> str:
        try:
            docs = self.retriever.invoke(query)
            if not docs:
                return "No information found"
            return "\n".join(doc.page_content for doc in docs)
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return f"Error during search: {str(e)}"

class PDFSearchTool(BaseTool):
    name: str = "PDF Search Tool"
    description: str = "Search information from the loaded PDF content."
    pdf_search_system: Any = Field(default=None, exclude=True)
    
    def __init__(self, pdf_search_system: PDFSearchSystem, **data):
        super().__init__(**data)
        self.pdf_search_system = pdf_search_system
    
    def _run(self, query: str) -> str:
        if isinstance(query, dict):
            query = query.get('query', '') if isinstance(query.get('query'), str) else str(query)
        return self.pdf_search_system.search_info(query)

def setup_pdf_search_system(pdf_content: str):
    pdf_system = PDFSearchSystem(pdf_content)
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

class QuestionData(BaseModel):
    question: str

def extract_text_from_pdf(pdf_file: BytesIO) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to process PDF file")

@pdf_router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        global global_pdf_content
        pdf_content = await file.read()
        pdf_file = BytesIO(pdf_content)
        pdf_text = extract_text_from_pdf(pdf_file)
        
        # Store in global variable
        global_pdf_content = pdf_text
        
        return {
            "message": "PDF processed successfully"
        }
    
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()

@pdf_router.post("/pdfrag")
async def pdf_rag_function(data: QuestionData):
    try:
        global global_pdf_content
        if not global_pdf_content:
            raise HTTPException(status_code=400, detail="Please upload a PDF first")
            
        agent = setup_pdf_search_system(global_pdf_content)
        result = search_pdf(agent, data.question)
        
        if isinstance(result, dict):
            if 'raw' in result:
                content = result['raw']
            elif 'tasks_output' in result and result['tasks_output']:
                content = result['tasks_output'][0].get('raw', '')
            else:
                content = "No content found"
        else:
            content = str(result)

        return {
            "content": content.strip()
        }
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))