#!/usr/bin/env python3
"""
Codey AI v5 - Unified FastAPI backend
Features:
 - Grok integration (XAI_API_KEY env)
 - NLP: spaCy (preferred) + transformers pipelines (summarize, sentiment, ner fallback)
 - Math via SymPy
 - Unit conversion
 - Code review via ast
 - MLP demo classifier
 - DuckDuckGo search
 - ChromaDB persistent memory + JSON knowledge base (entity memory + learned facts)
 - Emotion-aware tone (sentiment modifies style)
 - Render-ready: use uvicorn api:app --host 0.0.0.0 --port $PORT
"""

import os
import json
import re
import threading
import ast
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import numpy as np
from sympy import Eq, solve, simplify, sympify
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv

# Try to import chromadb, spacy, transformers — fail gracefully if unavailable
try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None

try:
    import spacy
except Exception:
    spacy = None

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
except Exception:
    pipeline = None

# local environment
load_dotenv()

# disable problematic ChromaDB deps (safe defaults)
os.environ.setdefault("CHROMA_DISABLE_ONNX", "1")
os.environ.setdefault("CHROMA_DISABLE_PULSAR", "1")

# --------------------- Pydantic request/response ---------------------
class AskRequest(BaseModel):
    question: str
    chat_history: Optional[List[dict]] = []
    user_id: Optional[str] = "default"

class AskResponse(BaseModel):
    answer: str
    source: str
    learned: bool
    tone: Optional[str] = None
    metadata: Optional[dict] = {}

# --------------------- KnowledgeBase (JSON) ---------------------
class KnowledgeBase:
    def __init__(self, memory_file: str = "codey_memory.json"):
        self.path = Path(memory_file)
        self._lock = threading.Lock()
        self.data = self._load()

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self):
        with self._lock:
            tmp = self.path.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            tmp.replace(self.path)

    def get(self, key: str):
        return self.data.get(key.lower())

    def learn(self, key: str, value):
        self.data[key.lower()] = value
        self._save()

    def dump(self) -> dict:
        return self.data

# --------------------- UnitConverter ---------------------
class UnitConverter:
    LENGTH = {
        ("m", "km"): lambda v: v/1000, ("km", "m"): lambda v: v*1000,
        ("miles", "km"): lambda v: v*1.60934, ("km", "miles"): lambda v: v/1.60934,
        ("m", "miles"): lambda v: v/1609.34, ("feet", "m"): lambda v: v*0.3048,
        ("m", "feet"): lambda v: v/0.3048,
    }
    MASS = {
        ("kg", "lb"): lambda v: v*2.20462, ("lb", "kg"): lambda v: v/2.20462,
        ("g", "kg"): lambda v: v/1000, ("kg", "g"): lambda v: v*1000,
    }

    def convert(self, text: str):
        m = re.search(r"([-+]?\d*\.?\d+)\s*([a-zA-Z°]+)\s+(?:to|in|->)\s+([a-zA-Z°]+)", text, re.I)
        if not m:
            return None
        try:
            val = float(m.group(1))
        except Exception:
            return None
        frm = m.group(2).lower().replace("°","")
        to = m.group(3).lower().replace("°","")

        if frm in ("c","celsius") and to in ("f","fahrenheit"):
            return f"{val}°C → {(val*9/5)+32:.2f}°F"
        if frm in ("f","fahrenheit") and to in ("c","celsius"):
            return f"{val}°F → {(val-32)*5/9:.2f}°C"

        for d in (self.LENGTH, self.MASS):
            key = (frm, to)
            if key in d:
                return f"{val} {frm} ≈ {d[key](val):.4g} {to}"
        return None

# --------------------- MathEngine ---------------------
class MathEngine:
    def solve_expression(self, expr: str):
        expr = expr.strip().replace("^", "**")
        try:
            if "=" in expr:
                left, right = map(str.strip, expr.split("=", 1))
                eq = Eq(sympify(left), sympify(right))
                sol = solve(eq)
                return f"Solution: {sol}"
            else:
                return f"Simplified: {simplify(sympify(expr))}"
        except Exception as e:
            return f"Math error: {e}"

# --------------------- CodeReviewer ---------------------
class CodeReviewer:
    def review(self, text: str):
        m = re.search(r"```(?:python)?\s*([\s\S]+?)```", text, re.I)
        if m:
            code = m.group(1)
        elif "def " in text or "class " in text:
            code = text
        else:
            return None
        try:
            tree = ast.parse(code)
            issues = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if ast.get_docstring(node) is None:
                        issues.append(f"- Add docstring for `{node.name}`")
                if isinstance(node, ast.ExceptHandler) and node.type is None:
                    issues.append("- Avoid bare `except:`")
            return "Code review:\n" + ("\n".join(issues) if issues else "No issues.")
        except Exception:
            return "Could not parse code. Wrap in ```python ... ```."

# --------------------- MLPModule (demo) ---------------------
class MLPModule:
    def __init__(self):
        X = np.array([[0.1,0.2,0.9],[0.9,0.8,0.1],[0.2,0.3,0.8],[0.8,0.7,0.2]])
        y = np.array([0,1,0,1])
        self.scaler = MinMaxScaler()
        Xs = self.scaler.fit_transform(X)
        self.model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=2000, random_state=42)
        self.model.fit(Xs, y)

    def predict(self, features):
        try:
            arr = np.asarray(features, dtype=float).reshape(1, -1)
            arr = self.scaler.transform(arr)
            return "Safe" if self.model.predict(arr)[0] == 0 else "Warning"
        except Exception:
            return "Invalid features"

# --------------------- FactRetriever ---------------------
class FactRetriever:
    FACTS = {
        "python": "Python: high-level programming language widely used in web dev, data science, and automation.",
        "ai": "AI simulates aspects of human intelligence with machines.",
        "mlp": "MLP is a feedforward neural network.",
        "sympy": "SymPy is a Python library for symbolic math.",
    }
    def get_fact(self, text: str):
        m = re.search(r"(?:what is|define|explain)\s+(.+?)(?:\?|$)", text, re.I)
        if m and (t := m.group(1).strip().lower()) in self.FACTS:
            return f"Fact: {self.FACTS[t]}"
        return None

# --------------------- GrokAI wrapper ---------------------
class GrokAI:
    API_KEY = os.getenv("XAI_API_KEY")
    MODEL = os.getenv("GROK_MODEL", "grok-4")
    ENDPOINT = os.getenv("GROK_ENDPOINT", "https://api.x.ai/v1/chat/completions")

    @classmethod
    def query(cls, prompt: str, timeout: int = 8) -> str:
        if not cls.API_KEY:
            return "XAI_API_KEY not set"
        try:
            r = requests.post(
                cls.ENDPOINT,
                json={
                    "model": cls.MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500
                },
                headers={"Authorization": f"Bearer {cls.API_KEY}"},
                timeout=timeout
            )
            r.raise_for_status()
            data = r.json()
            # safe extraction
            return data.get("choices", [{}])[0].get("message", {}).get("content", "") or str(data)
        except Exception as e:
            return f"Grok error: {e}"

# --------------------- NLPProcessor ---------------------
class NLPProcessor:
    def __init__(self):
        self.use_spacy = False
        self.use_transformers = False
        self._load_models()

    def _load_models(self):
        # spaCy (preferred)
        try:
            if spacy is not None:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    self.use_spacy = True
                except Exception:
                    # model not installed — user should run: python -m spacy download en_core_web_sm
                    self.nlp = None
                    self.use_spacy = False
        except Exception:
            self.nlp = None
            self.use_spacy = False

        # transformers pipelines (summarization, sentiment, ner fallback)
        if pipeline is not None:
            try:
                # summarizer: choose a lighter model if available
                try:
                    self.summarizer = pipeline("summarization", model=os.getenv("SUM_MODEL","sshleifer/distilbart-cnn-12-6"))
                except Exception:
                    self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                self.sentiment = pipeline("sentiment-analysis")
                # NER fallback
                self.ner = pipeline("ner", aggregation_strategy="simple")
                self.use_transformers = True
            except Exception:
                self.summarizer = None
                self.sentiment = None
                self.ner = None
                self.use_transformers = False
        else:
            self.summarizer = None
            self.sentiment = None
            self.ner = None
            self.use_transformers = False

    def analyze(self, text: str) -> Dict[str, Any]:
        entities = []
        noun_chunks = []
        try:
            if self.use_spacy and self.nlp:
                doc = self.nlp(text)
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                noun_chunks = [chunk.text for chunk in doc.noun_chunks]
            elif self.use_transformers and self.ner:
                ner_out = self.ner(text)
                entities = [(r.get("word"), r.get("entity_group")) for r in ner_out]
            else:
                entities = []
                noun_chunks = []
        except Exception:
            entities = []
            noun_chunks = []
        return {"entities": entities, "topics": noun_chunks}

    def summarize(self, text: str) -> str:
        if not self.summarizer:
            return "Summarization not available (install transformers and a summarization model)."
        try:
            # guard for short text
            if len(text.split()) < 20:
                return text if len(text.split()) < 80 else text[:500] + "..."
            out = self.summarizer(text, max_length=80, min_length=25, do_sample=False)
            return out[0]["summary_text"] if isinstance(out, list) else str(out)
        except Exception as e:
            return f"Summarization error: {e}"

    def sentiment_label(self, text: str) -> Tuple[str, float]:
        if not self.sentiment:
            return ("neutral", 0.0)
        try:
            res = self.sentiment(text)[0]
            return (res.get("label", "neutral").lower(), float(res.get("score", 0.0)))
        except Exception:
            return ("neutral", 0.0)

# --------------------- LearningMemory (Chroma + entity memory) ---------------------
class LearningMemory:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.enabled = False
        self.collection = None
        self.embedder = None
        # JSON fallback for simple storage
        self.kb = KnowledgeBase("codey_memory.json")

        if chromadb is None:
            print("chromadb not installed — falling back to JSON-only memory.")
            self.enabled = False
            return

        try:
            # instantiate persistent client
            # Using path param for persistent storage
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            # create/get collection
            self.collection = self.client.get_or_create_collection(name="codey_memory")
            # try to use sentence-transformers via langchain_community if available
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                self.embedder = HuggingFaceEmbeddings(model_name=os.getenv("EMBED_MODEL","sentence-transformers/all-MiniLM-L6-v2"))
            except Exception:
                self.embedder = None
            self.enabled = True
        except Exception as e:
            print("Chroma init failed:", e)
            self.enabled = False

    def _embed(self, text: str):
        if self.embedder:
            try:
                return self.embedder.embed_query(text)
            except Exception:
                pass
        # fallback: simple hash-based vector (not ideal but avoids crash)
        h = hash(text)
        # small deterministic vector
        vec = [((h >> i) & 0xFF) / 255.0 for i in range(0, 128)]
        return vec

    def add_knowledge(self, question: str, answer: str, source: str = "web"):
        # add JSON memory
        self.kb.learn(question, {"answer": answer, "source": source})
        if not self.enabled or self.collection is None:
            return
        try:
            doc = f"Q: {question}\nA: {answer}\nSource: {source}"
            emb = self._embed(doc)
            doc_id = f"doc_{abs(hash(doc)) % 10000000}"
            self.collection.add(embeddings=[emb], documents=[doc], ids=[doc_id])
        except Exception as e:
            print("Memory add error:", e)

    def query_memory(self, question: str) -> Optional[str]:
        # check JSON KB first (exact)
        got = self.kb.get(question)
        if got:
            return got.get("answer") if isinstance(got, dict) else got
        if not self.enabled or self.collection is None:
            return None
        try:
            emb = self._embed(question)
            results = self.collection.query(query_embeddings=[emb], n_results=1)
            docs = results.get("documents", [[]])
            if docs and docs[0]:
                return docs[0][0]
        except Exception as e:
            print("Memory query error:", e)
        return None

    def store_entities(self, user_id: str, entities: List[Tuple[str, str]]):
        # persist entity mentions into JSON KB grouped by user
        if not entities:
            return
        key = f"entities:{user_id}"
        old = self.kb.get(key) or []
        # convert tuples to dicts
        new = [{"text": e[0], "label": e[1]} for e in entities if e]
        merged = old + new if isinstance(old, list) else new
        self.kb.learn(key, merged)

# --------------------- Search Tool ---------------------
try:
    from langchain_community.tools import DuckDuckGoSearchRun
    search = DuckDuckGoSearchRun()
except Exception:
    # simple web-search placeholder (no network scraping)
    search = None

# --------------------- Prompt / Behavior ---------------------
HUMAN_PROMPT = """You are Codey, a helpful assistant. Be clear, concise, and friendly.
SentimentTone: {tone}
Context: {chat_history}
Question: {question}
"""

# --------------------- Brain ---------------------
class CodeyBrain:
    def __init__(self):
        self.kb = KnowledgeBase()
        self.memory = LearningMemory()
        self.units = UnitConverter()
        self.math = MathEngine()
        self.reviewer = CodeReviewer()
        self.mlp = MLPModule()
        self.facts = FactRetriever()
        self.nlp = NLPProcessor()
        self.simple = {
            "hello": "Hey! What's up? Spill — what's on your mind today?",
            "hi": "Yo! Codey here, ready to geek out or just chat."
        }

    def _tone_from_sentiment(self, label: str, score: float) -> str:
        # map sentiment to a simple tone label
        if label.lower() in ("positive", "pos", "joy", "happy") and score > 0.6:
            return "upbeat"
        if label.lower() in ("negative", "neg", "sad", "anger") and score > 0.6:
            return "concise-calm"
        return "neutral-friendly"

    def _style_prefix_for_tone(self, tone: str) -> str:
        if tone == "upbeat":
            return "😊 "
        if tone == "concise-calm":
            return "🤝 "
        return ""

    def respond(self, text: str, chat_history: List[dict], user_id: str) -> Tuple[str,str,bool,dict]:
        txt = (text or "").strip()
        low = txt.lower()
        learned = False
        metadata = {}

        # 1) easy greetings
        if low in self.simple:
            return self.simple[low], "local", False, {}

        # 2) memory (vector/JSON)
        mem_ans = self.memory.query_memory(txt)
        if mem_ans:
            return f"I recall: {mem_ans}", "learned", False, {}

        # 3) unit conversion
        if (ans := self.units.convert(txt)):
            return f"{ans} – easy!", "units", False, {}

        # 4) math detection
        if re.search(r"\d\s*[\+\-\*/\^=]", txt):
            return f"{self.math.solve_expression(txt)} Math done!", "math", False, {}

        # 5) code review
        if (ans := self.reviewer.review(txt)):
            return f"{ans} Let's fix it?", "code", False, {}

        # 6) facts
        if (ans := self.facts.get_fact(txt)):
            return f"{ans} Fun fact!", "fact", False, {}

        # 7) predict => mlp
        if "predict" in low:
            nums = re.findall(r"[-+]?\d*\.?\d+", txt)
            if len(nums) >= 3:
                return f"{self.mlp.predict([float(n) for n in nums[:3]])} – ML says!", "mlp", False, {}

        # 8) NLP commands: summarize, analyze, sentiment
        if "summarize" in low or "summary" in low:
            # remove the trigger word and summarize the rest
            raw = re.sub(r"(summarize|summary of)\s*", "", txt, flags=re.I).strip()
            if not raw:
                return "What text would you like summarized?", "nlp", False, {}
            summary = self.nlp.summarize(raw)
            tone_label, tone_score = self.nlp.sentiment_label(raw)
            tone = self._tone_from_sentiment(tone_label, tone_score)
            prefix = self._style_prefix_for_tone(tone)
            metadata = {"sentiment": {"label": tone_label, "score": tone_score}}
            return f"{prefix}{summary}", "nlp-summary", False, metadata

        if "analyze" in low or "topics" in low or "entities" in low:
            analysis = self.nlp.analyze(txt)
            # store entities in memory for the user
            self.memory.store_entities(user_id, analysis.get("entities", []))
            return f"Entities: {analysis.get('entities')} | Topics: {analysis.get('topics')}", "nlp-analysis", False, {}

        if "sentiment" in low:
            label, score = self.nlp.sentiment_label(txt)
            tone = self._tone_from_sentiment(label, score)
            prefix = self._style_prefix_for_tone(tone)
            return f"{prefix}Sentiment: {label} (confidence {score:.2f})", "nlp-sentiment", False, {"sentiment": {"label": label, "score": score}}

        # 9) check local KB exact match
        if (ans := self.kb.get(low)):
            return ans, "kb", False, {}

        # 10) use Grok via prompt
        tone_label, tone_score = self.nlp.sentiment_label(txt)
        tone = self._tone_from_sentiment(tone_label, tone_score)
        prefix = self._style_prefix_for_tone(tone)
        prompt = HUMAN_PROMPT.format(question=txt, chat_history=json.dumps(chat_history), tone=tone)
        grok_ans = GrokAI.query(prompt)

        # 11) if Grok indicates "search" or is unsure, perform web search and store
        grok_lower = (grok_ans or "").lower()
        if ("search" in grok_lower or "don't know" in grok_lower or "i'm not sure" in grok_lower) and search is not None:
            try:
                result = search.run(txt)
                analyzed = GrokAI.query(f"Rephrase this search casually: {result}\nFor: {txt}")
                self.memory.add_knowledge(txt, analyzed, "web")
                self.kb.learn(txt, analyzed)
                learned = True
                return f"{prefix}{analyzed} (Learned!)", "search", learned, {"source": "web"}
            except Exception as e:
                # fallback: return Grok answer
                return f"{prefix}{grok_ans}", "grok", False, {"error": str(e)}

        # 12) fallback return Grok answer (with tone prefix)
        return f"{prefix}{grok_ans}", "grok", learned, {}

# --------------------- FastAPI app ---------------------
app = FastAPI(title="Codey AI v5", version="5.0")
brain = CodeyBrain()

@app.get("/health")
async def health():
    return {"status": "ok", "chromadb": bool(chromadb), "spacy": bool(spacy), "transformers": bool(pipeline)}

@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(req: AskRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="No question provided.")
    answer, source, learned, metadata = brain.respond(req.question, req.chat_history or [], req.user_id or "default")
    return AskResponse(answer=answer, source=source, learned=learned, tone=metadata.get("tone"), metadata=metadata)

@app.get("/memory")
async def dump_memory():
    # return JSON memory and chroma availability
    return {"kb": brain.kb.dump(), "chroma_enabled": brain.memory.enabled}

# Create DB directory if missing
Path("./chroma_db").mkdir(exist_ok=True)

