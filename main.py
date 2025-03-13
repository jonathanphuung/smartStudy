from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

app = FastAPI()

# Load the summarization model from Hugging Face
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Data model for handling request body
class TextRequest(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "Welcome to the Smart Study Planner API"}

@app.post("/summarize/")
async def summarize(request: BaseModel):
    summary = summarizer(request.text, max_length=130, min_length=30, do_sample=False)
    return {"summary": summary[0]['summary_text']}