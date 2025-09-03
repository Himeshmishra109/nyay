from flask import Flask, render_template, request, jsonify
from nyay_pipeline import NyayProPipeline
import os

app = Flask(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "nyay_pro_app", "data", "legal_texts")
pipeline = NyayProPipeline(data_dir=DATA_DIR)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    q = request.form.get("question", "").strip()
    lang = request.form.get("lang", "en").strip().lower()
    explain = request.form.get("explain", "true").lower() in ("1","true","yes")
    if not q:
        return jsonify({"ok": False, "error": "Please enter a question."}), 400
    result = pipeline.answer(q, user_lang=lang, explain=explain)
    return jsonify({"ok": True, **result})

if __name__ == "__main__":
    app.run(debug=True)
