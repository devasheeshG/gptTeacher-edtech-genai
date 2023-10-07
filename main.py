from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from utils.embed_pdf_into_vectordb import embed_pdf
import os

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    """Checks if the file is allowed to be uploaded"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    """Displays the index page accessible at '/'"""
    return "Welcome to GPT-Teacher API"


@app.route("/upload", methods=["GET"])
def upload():
    """Uploads a file to the server"""

    file = request.files["file"]

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        
        status = embed_pdf(filepath)
        
        if status is not True:
            return jsonify({"error": "Something went wrong"}), 400
        
        os.remove(filepath)
        return jsonify({"message": "File uploaded and embeded into vector Database Successfully"}), 200
    else:
        return (
            jsonify({"error": "Invalid file format. Only PDF files are allowed."}),
            400,
        )



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8002, threaded=True, use_reloader=False)
