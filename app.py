import os
import json
import re
from flask import Flask, request, jsonify, send_from_directory
from groq import Groq

app = Flask(__name__, static_folder=".")
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def plain_text_to_words(text):
    words = []
    for line in text.splitlines():
        for word in line.split():
            words.append({"text": word, "confidence": 85, "guess": None})
        words.append({"newline": True})
    return words


def parse_response(raw):
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip()).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "words" in parsed:
            return parsed
        for key in ("transcription", "text", "content", "result"):
            if key in parsed and isinstance(parsed[key], str):
                return {"words": plain_text_to_words(parsed[key]), "overall_confidence": 80}
        if isinstance(parsed, list):
            return {"words": parsed, "overall_confidence": 80}
    except json.JSONDecodeError:
        pass

    match = re.search(r'\{[\s\S]*\}', cleaned)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, dict) and "words" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    if cleaned:
        return {"words": plain_text_to_words(cleaned), "overall_confidence": 75}

    return None


def words_to_plain(words):
    """Reconstruct plain text from words array for the cleanup pass."""
    out = []
    for token in words:
        if token.get("newline"):
            out.append("\n")
        else:
            w = token.get("guess") or token.get("text", "")
            out.append(w)
    return " ".join(out).replace(" \n ", "\n").strip()


def cleanup_pass(words):
    """
    Second LLM pass: give the model the raw transcription and ask it to
    fix obvious errors using language understanding, then re-score confidence.
    Returns an updated words array.
    """
    plain = words_to_plain(words)
    if not plain.strip():
        return words

    prompt = f"""You are a post-processing assistant for a handwriting OCR system.

Below is a raw transcription that may contain errors from difficult handwriting.
Your job is to fix likely misread words using:
- Common English words and phrases
- Grammar and sentence structure
- Context from surrounding words
- Typical handwriting confusion patterns (e.g. "m"↔"n", "u"↔"v"↔"w", "i"↔"l"↔"1", "o"↔"0", "rn"↔"m", "cl"↔"d", "li"↔"h")

RAW TRANSCRIPTION:
{plain}

Return ONLY this JSON, no explanation:
{{
  "words": [
    {{"text": "corrected_word", "confidence": 92, "guess": null}},
    {{"text": "uncertain_word", "confidence": 48, "guess": "best_guess"}},
    {{"newline": true}}
  ],
  "overall_confidence": 85
}}

Rules:
- Re-evaluate every word's confidence after correction
- If you corrected a word, set confidence to reflect remaining uncertainty
- If a word is still ambiguous after correction, confidence < 70 and provide "guess"
- Preserve original line breaks with {{"newline": true}}
- Output ONLY raw JSON"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # text-only model, stronger language reasoning
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
        temperature=0.1,
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content
    result = parse_response(raw)
    return result if result else {"words": words, "overall_confidence": 75}


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

    # ── Pass 1: Vision model reads the handwriting ──────────────────────────
    vision_prompt = """You are an expert forensic handwriting analyst. Examine this handwritten document with extreme care.

Your task: transcribe EVERY word, letter by letter if needed.

Common handwriting traps to watch for:
- "rn" looks like "m", "cl" looks like "d", "li" looks like "h" or "b"
- "u/v/w" are often confused, so are "i/l/1", "o/0", "n/u"
- Cursive letters blend together — look at word length and ascenders/descenders
- Short words: "the", "and", "of", "to", "in", "is", "it", "be", "as", "at"
- Look at word shape/envelope, not just individual letters

Return ONLY this JSON:
{
  "words": [
    {"text": "Hello", "confidence": 95, "guess": null},
    {"text": "wrld", "confidence": 40, "guess": "world"},
    {"newline": true}
  ],
  "overall_confidence": 82
}

- One object per word. {"newline": true} for line breaks.
- confidence 0-100: visual legibility + contextual fit combined
- confidence < 70 → set "guess" to best prediction, else "guess" is null
- Punctuation stays with its word. Output ONLY raw JSON."""

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                {"type": "text", "text": vision_prompt}
            ]
        }],
        max_tokens=4096,
        temperature=0.05,
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content
    vision_result = parse_response(raw)

    if vision_result is None:
        return jsonify({"error": "Could not extract text from image"}), 500

    # ── Pass 2: Language model cleans up using grammar + context ────────────
    final_result = cleanup_pass(vision_result["words"])

    return jsonify(final_result)


if __name__ == "__main__":
    app.run(debug=True)
