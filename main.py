from fastapi import FastAPI
from transformers import pipeline
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# Load Hugging Face model (e.g., summarization or sentiment analysis)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.get("/")
async def root():
    return {"message": "Welcome to the Smart Study Planner API"}

@app.post("/summarize/")
async def summarize(text: str):
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return {"summary": summary[0]['summary_text']}