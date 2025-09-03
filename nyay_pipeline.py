import os
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import re

# Optional imports
_transformers_available = True
_googletrans_available = True
try:
    from transformers import pipeline as hf_pipeline
except Exception:
    _transformers_available = False

try:
    from googletrans import Translator
except Exception:
    _googletrans_available = False

def normalize_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text: str, min_chars=900, max_chars=1200):
    try:
        sents = sent_tokenize(text)
    except Exception:
        sents = re.split(r'(?<=[.?!])\s+', text)
    chunks = []
    buf = ""
    for s in sents:
        if not s.strip(): continue
        if len(buf) + len(s) + 1 <= max_chars:
            buf = (buf + " " + s).strip()
        else:
            if len(buf) < min_chars and len(s) < max_chars:
                buf = (buf + " " + s).strip()
            chunks.append(buf)
            buf = s
    if buf: chunks.append(buf)
    return [c.strip() for c in chunks if c.strip()]

# Simple explanation rewrite rules (basic)
SIMPLIFY_MAP = {
    "shall not": "must not",
    "shall": "must",
    "deprived": "denied",
    "without prejudice": "without affecting other rights",
    "hereinafter": "from now on",
    "thereof": "of that",
    "notwithstanding": "despite",
    "aforesaid": "previously mentioned",
    "liable": "responsible",
    "provision": "rule",
    "affidavit": "written statement",
    "custody": "keeping",
    "null and void": "invalid",
}

def simplify_text(text: str) -> str:
    out = text
    for k, v in SIMPLIFY_MAP.items():
        out = re.sub(r'\\b' + re.escape(k) + r'\\b', v, out, flags=re.IGNORECASE)
    # small template to make it more conversational
    return "In simple terms: " + out

class NyayProPipeline:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.chunks = []
        self.chunk_sources = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.qa_model = None
        self.translator = None
        self._load_and_build()
        self._init_optional_models()

    def _load_and_build(self):
        texts = []
        names = []
        for fn in sorted(os.listdir(self.data_dir)):
            if fn.lower().endswith('.txt'):
                path = os.path.join(self.data_dir, fn)
                with open(path, 'r', encoding='utf-8') as f:
                    raw = f.read()
                texts.append(normalize_text(raw))
                names.append(fn)
        chunks = []
        sources = []
        for name, text in zip(names, texts):
            cks = chunk_text(text, min_chars=900, max_chars=1200)
            for i, ck in enumerate(cks):
                chunks.append(ck)
                sources.append((name, i))
        self.chunks = chunks
        self.chunk_sources = sources
        if self.chunks:
            self.vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english', lowercase=True, max_df=0.5)
            self.tfidf_matrix = self.vectorizer.fit_transform(self.chunks)
        else:
            self.vectorizer = None
            self.tfidf_matrix = None

    def _init_optional_models(self):
        if _transformers_available:
            try:
                self.qa_model = hf_pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')
            except Exception:
                self.qa_model = None
        if _googletrans_available:
            try:
                self.translator = Translator()
            except Exception:
                self.translator = None

    def _top_chunks(self, query: str, k: int = 3):
        if not self.chunks:
            return []
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.tfidf_matrix)[0]
        top_idx = sims.argsort()[::-1][:k]
        return [(int(i), float(sims[i])) for i in top_idx]

    def _extract_sentences(self, chunk: str, query: str, max_sentences: int = 3) -> List[str]:
        try:
            sents = sent_tokenize(chunk)
        except Exception:
            sents = re.split(r'(?<=[.?!])\s+', chunk)
        if not sents:
            return [chunk]
        sent_vecs = self.vectorizer.transform(sents)
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, sent_vecs)[0]
        order = sims.argsort()[::-1]
        chosen = []
        for idx in order[:max_sentences]:
            chosen.append(sents[idx].strip())
        return chosen or [sents[0]]

    def _qa_with_model(self, question: str, context: str):
        if not self.qa_model:
            return None
        try:
            out = self.qa_model(question=question, context=context)
            if isinstance(out, dict):
                return out.get('answer', None)
            elif isinstance(out, list) and out:
                return out[0].get('answer', None)
            else:
                return None
        except Exception:
            return None

    def _translate_to_en(self, text: str):
        if not self.translator:
            return text
        try:
            return self.translator.translate(text, dest='en').text
        except Exception:
            return text

    def _translate_from_en(self, text: str, dest='hi'):
        if not self.translator:
            return text
        try:
            return self.translator.translate(text, dest=dest).text
        except Exception:
            return text

    def answer(self, query: str, user_lang: str = 'en', explain: bool = True) -> Dict:
        original_query = query
        lang = user_lang.lower() if user_lang else 'en'
        # translate query to English for processing if Hindi
        if lang.startswith('hi'):
            q_en = self._translate_to_en(query)
        else:
            q_en = query

        if not self.chunks:
            return {"answer": "No documents loaded. Please add .txt files to data/legal_texts.", "context": "", "source": ""}

        # retrieve top chunks
        tops = self._top_chunks(q_en, k=3)
        best_idx, best_score = tops[0]
        best_chunk = self.chunks[best_idx]
        name, cidx = self.chunk_sources[best_idx]

        # Attempt QA with model
        answer_text = None
        model_used = "tfidf-extractive"
        if self.qa_model:
            # try model on concatenated top chunks for more context
            combined = " ".join([self.chunks[i] for i, _ in tops])
            ans = self._qa_with_model(question=q_en, context=combined)
            if ans:
                answer_text = ans
                model_used = "distilbert-qa"

        # Fallback to extractive sentence scoring
        if not answer_text:
            sents = self._extract_sentences(best_chunk, q_en, max_sentences=3)
            answer_text = " ".join(sents)

        # Simple explanation rewrite (rule-based)
        explanation = simplify_text(answer_text) if explain else ""

        # Translate back if needed
        if lang.startswith('hi'):
            answer_out = self._translate_from_en(answer_text, dest='hi')
            explanation_out = self._translate_from_en(explanation, dest='hi') if explanation else ""
        else:
            answer_out = answer_text
            explanation_out = explanation

        return {
            "answer": answer_out,
            "explanation": explanation_out,
            "context": best_chunk,
            "source": f"{name} (chunk #{cidx+1})",
            "similarity": round(best_score, 4),
            "model_used": model_used,
            "original_question": original_query
        }
