from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

class SummarizeRequest(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "Welcome to the Smart Study Planner API"}

@app.post("/summarize/")
async def summarize(request: SummarizeRequest):
    summary = summarizer(request.text, max_length=130, min_length=30, do_sample=False)
    return {"summary": summary[0]['summary_text']}