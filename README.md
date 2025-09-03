# nyay_pro — Full-featured Legal QA (Designed to match your Problem & Solution)

**Goal:** A user-friendly system that loads unstructured legal texts (Constitution / law books), preprocesses them, uses TF-IDF retrieval with cosine similarity to find relevant chunks, and then provides precise, context-based answers using an optional pre-trained QA model. The app supports bilingual (English/Hindi) UI, voice input, TTS, and simple plain-language explanations for users.

**Design choices to keep it smooth:** Default behavior uses TF-IDF (fast). Optional advanced extractive QA (DistilBERT) and translation are enabled only if corresponding packages are installed; otherwise the app gracefully falls back to TF-IDF extractive answers and a lightweight explanation module.

## Features
- Load multiple `.txt` legal documents from `nyay_pro_app/data/legal_texts/`
- Preprocessing & chunking (~900-1200 char chunks)
- TF-IDF vector store + cosine similarity retrieval
- Optional transformer-based extractive QA (DistilBERT) for higher accuracy
- Simple explanation module that rewrites extracted answers into plain language (rule-based + template)
- Bilingual UI (English / हिंदी) with optional translation via `googletrans`
- Browser voice input and TTS
- Flask web interface (lightweight & mobile-friendly)
- Clear source attribution (file name + chunk number) and similarity score

## Quick start
1) Extract the ZIP and open terminal in the extracted folder.
2) Create & activate venv:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```
3) Install dependencies (core):
```bash
pip install -r requirements.txt
```
To enable advanced QA & translation, ensure `transformers`, `torch`, and `googletrans` are installed after core install (they are optional and large).

4) Run:
```bash
python app.py
```

Open: http://127.0.0.1:5000/

## Notes
- Add any large legal `.txt` files to `nyay_pro_app/data/legal_texts/`.
- Advanced QA requires internet for model download on first run, and sufficient disk and memory.
- For production, consider storing vector embeddings in a persistent store (FAISS, Milvus) and fine-tuning a legal-domain model.

---
This package was built to closely follow the Problem Statement and Proposed Solution while keeping the app runnable on modest machines.
