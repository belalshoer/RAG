from flask import Flask, request, jsonify
import os
from typing import Dict
from flask import render_template


from rag_test import DestinationQA 

app = Flask(__name__)

PDF_PATH = os.getenv("PDF_PATH", "salka_routes.pdf")
DB_DIR = os.getenv("DB_DIR", "./chroma_db")

qa_system = None

def initialize_qa_system():
    global qa_system
    try:
        qa_system = DestinationQA(PDF_PATH, DB_DIR)
        qa_system.initialize_qa_system()
        print("QA system initialized successfully")
    except Exception as e:
        print(f"Failed to initialize QA system: {e}")
        raise

initialize_qa_system()
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def query_destinations():
    if not qa_system:
        return jsonify({"error": "QA system not initialized"}), 500

    try:

        if request.is_json:
            data = request.get_json()
            question = data.get('question', '').strip()
        else:
            question = request.form.get('question', '').strip()

        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400

        result = qa_system.query_destinations(question)
        return jsonify({
            "answer": result["answer"],
            "sources": result["sources"],
            "error": ""
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "answer": f"Sorry, an error occurred: {str(e)}",
            "sources": []
        }), 500
    
# @app.route('/')
# def home():
#     return "Destination QA System is running. Use /api/query to ask questions."

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if qa_system else "uninitialized",
        "pdf_loaded": os.path.exists(PDF_PATH) if PDF_PATH else False,
        "vector_db_exists": os.path.exists(DB_DIR) if DB_DIR else False
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)