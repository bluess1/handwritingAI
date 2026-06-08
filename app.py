import os
import json
import re
from flask import Flask, request, jsonify, send_from_directory
from groq import Groq

app = Flask(__name__, static_folder=".")
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def plain_text_to_words(text):
    """Convert a plain transcription string into the words array format."""
    words = []
    for line in text.splitlines():
        for word in line.split():
            words.append({"text": word, "confidence": 85, "guess": None})
        words.append({"newline": True})
    return words

def parse_response(raw):
    """Try every strategy to get a valid words array out of the model response."""

    # 1. Strip markdown fences
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip())
    cleaned = cleaned.strip()

    # 2. Try direct JSON parse
    try:
        parsed = json.loads(cleaned)

        # Has the right shape already
        if isinstance(parsed, dict) and "words" in parsed:
            return parsed

        # Model returned {"transcription": "..."} or {"text": "..."}
        for key in ("transcription", "text", "content", "result"):
            if key in parsed and isinstance(parsed[key], str):
                return {
                    "words": plain_text_to_words(parsed[key]),
                    "overall_confidence": 80
                }

        # Model returned a list directly
        if isinstance(parsed, list):
            return {"words": parsed, "overall_confidence": 80}

    except json.JSONDecodeError:
        pass

    # 3. Try to extract a JSON object from somewhere in the text
    match = re.search(r'\{[\s\S]*\}', cleaned)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict) and "words" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    # 4. Final fallback: treat entire response as plain text transcription
    if cleaned:
        return {
            "words": plain_text_to_words(cleaned),
            "overall_confidence": 75
        }

    return None


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

    prompt = """Transcribe every word of handwritten text visible in this image.

Respond with ONLY this JSON — no explanation, no markdown:
{
  "words": [
    {"text": "Hello", "confidence": 95, "guess": null},
    {"text": "wrld", "confidence": 45, "guess": "world"},
    {"newline": true}
  ],
  "overall_confidence": 82
}

Rules:
- One object per word. Use {"newline": true} for line breaks.
- confidence: 0-100. Score based on visual legibility AND whether the word fits the sentence.
- If confidence < 70, set "guess" to your best prediction. Otherwise "guess" is null.
- Punctuation stays attached to its word."""

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                {"type": "text", "text": prompt}
            ]
        }],
        max_tokens=4096,
        temperature=0.1,
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content

    result = parse_response(raw)

    if result is None:
        return jsonify({"error": "Could not extract text from image", "raw": raw}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
