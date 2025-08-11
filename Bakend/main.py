from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import joblib
from transformers import AutoTokenizer
import os
from model import BertWithTFIDF

app = FastAPI()

# 允许跨域（根据需要调整）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发阶段方便，生产建议限定具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

# 全局变量
model = None
tokenizer = None
tfidf_vectorizer = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextInput(BaseModel):
    text: str

@app.on_event("startup")
def load_models():
    global model, tokenizer, tfidf_vectorizer
    print("🔧 Loading model, tokenizer, and TF-IDF vectorizer...")

    tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    tokenizer = AutoTokenizer.from_pretrained("models/tokenizer")
    model_path = "models/bert_tfidf_fusion_model.pt"
    model = BertWithTFIDF("roberta-large", tfidf_dim=1000, num_labels=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # NLP project 目录
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "frontend"))

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict/")
async def predict(input: TextInput):
    text = input.text

    tfidf_feat = tfidf_vectorizer.transform([text]).toarray()
    tfidf_tensor = torch.tensor(tfidf_feat, dtype=torch.float).to(device)

    encoded = tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask, tfidf_tensor)
        pred_class = torch.argmax(logits, dim=1).item()

    return {"prediction": pred_class}
