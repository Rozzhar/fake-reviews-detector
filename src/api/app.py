from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "API para detector de reseñas falsas"}
