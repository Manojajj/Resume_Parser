import warnings
from flask import Flask, request, jsonify, render_template
from model import parse_resume  # Import your parsing logic

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Render the upload form

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'resume' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['resume']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file and file.filename.endswith('.pdf'):
        resume_text = parse_resume(file)  # Parse the resume
        return jsonify(resume_text)  # Return the parsed data in JSON format
    else:
        return jsonify({"error": "Only PDF files are allowed"}), 400

if __name__ == "__main__":
    app.run(debug=True)
