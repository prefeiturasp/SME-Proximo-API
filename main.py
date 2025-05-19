from fastapi import FastAPI
from routers.api import router as api_router

app = FastAPI(
    title="API Adaptativa",
    description="API para testes adaptativos com parâmetros complexos",
    version="1.0.0"
)

app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    