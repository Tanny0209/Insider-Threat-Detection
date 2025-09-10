from fastapi import FastAPI

app = FastAPI(title="Internal Eye - Backend")

@app.get("/health")
async def health():
    return {"status": "ok"}
