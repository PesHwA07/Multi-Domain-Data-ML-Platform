from fastapi import FastAPI

app = FastAPI(title="Multi-Domain Data & ML Platform API")

@app.get("/health")
def health_check():
