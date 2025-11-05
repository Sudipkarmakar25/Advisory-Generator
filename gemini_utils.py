import os
import json
import pandas as pd
import google.generativeai as genai
from logger_utils import log_event
from retrain_utils import retrain_model

DATA_FILE = "sample_data.csv"

# Set API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyAk1L0TlG_SjIZTqztHSQENnILvD-nVmEI"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

def get_gemini_model():
    models = [
        "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite",
        "models/gemini-2.5-flash", "models/gemini-2.5-pro", "models/gemini-2.5-flash-lite"
    ]
    for m in models:
        try:
            model = genai.GenerativeModel(m)
            _ = model.count_tokens("test")
            log_event(f"Gemini model ready: {m}")
            return model
        except Exception:
            continue
    raise RuntimeError("No working Gemini model found.")

GEMINI_MODEL = get_gemini_model()

def call_gemini_fallback(record):
    prompt = f"""
You are an expert agronomist. Analyze this crop data and give JSON with crop label and advice.
Crop: {record['crop_name']}
Location: {record['location']}
Weather: {record.get('weather', 'unknown')}
Soil Type: {record['soiltype']}
Temperature: {record['temperature']}°C
Humidity: {record['humidity']}%
Rainfall: {record['rainfall']} mm

Respond only in JSON:
{{ "label": "healthy"|"moderate"|"stress", "suggestion": "short advice" }}
"""
    try:
        response = GEMINI_MODEL.generate_content(prompt)
        text = response.text.strip()
        if "{" in text and "}" in text:
            text = text[text.index("{"): text.rindex("}") + 1]
        data = json.loads(text)
        label = data.get("label", "moderate")
        suggestion = data.get("suggestion", "Keep observing field conditions.")
    except Exception as e:
        log_event(f"Gemini parse error: {e}")
        label = "moderate"
        suggestion = "Monitor the crop manually; fallback occurred."

    new_entry = {
        "crop_name": record["crop_name"],
        "location": record["location"],
        "weather": record.get("weather", "unknown"),
        "soiltype": record["soiltype"],
        "temperature": record["temperature"],
        "humidity": record["humidity"],
        "rainfall": record["rainfall"],
        "label": label
    }

    df = pd.read_csv(DATA_FILE)
    exists = (
        (df["crop_name"] == new_entry["crop_name"]) &
        (df["location"] == new_entry["location"]) &
        (df["temperature"] == new_entry["temperature"]) &
        (df["humidity"] == new_entry["humidity"]) &
        (df["rainfall"] == new_entry["rainfall"])
    ).any()

    if not exists:
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)
        log_event(f"New data added via Gemini fallback: {new_entry['location']} ({label})")
        retrain_model()
        log_event("Model retrained after fallback")
    else:
        log_event(f"Duplicate record for {new_entry['location']} skipped")

    return label, suggestion
