from fastapi import FastAPI
from app.core.config import settings
from app.api.pdf_router import pdf_router
from app.utils.logger import setup_logging
from app.middleware.cors import add_cors_middleware

def include_router(app):
    app.include_router(
        pdf_router,
        prefix="/api",
        tags=["PDF RAG Endpoints"],
    )

def start_application():
    app = FastAPI(title=settings.PROJECT_NAME, version=settings.PROJECT_VERSION)
    setup_logging()
    add_cors_middleware(app)
    include_router(app)
    return app

app = start_application()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", reload=True)
