import os
import base64
from flask import Flask, request, jsonify, send_from_directory
from groq import Groq

app = Flask(__name__, static_folder=".")
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/decode", methods=["POST"])
def decode():
    data = request.json
    image_b64 = data.get("image")
    mime = data.get("mime", "image/jpeg")

    if not image_b64:
        return jsonify({"error": "No image provided"}), 400

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                {"type": "text", "text": "Transcribe all handwritten text in this image exactly as written. Preserve line breaks and structure. Mark unclear words as [unclear]. Output only the transcribed text, nothing else."}
            ]
        }],
        max_tokens=2048,
        temperature=0.1
    )

    return jsonify({"transcription": response.choices[0].message.content})

if __name__ == "__main__":
    app.run(debug=True)
