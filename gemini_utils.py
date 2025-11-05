import os
import json
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from logger_utils import log_event
from retrain_utils import retrain_model


load_dotenv()  

DATA_FILE = os.getenv("DATA_FILE", "sample_data.csv")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise EnvironmentError("❌ GOOGLE_API_KEY not found. Add it to your .env file.")


try:
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    raise RuntimeError(f"❌ Gemini configuration failed: {e}")



def get_gemini_model():
    """Try multiple Gemini models until one works."""
    models = [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "models/gemini-2.5-pro",
        "models/gemini-2.5-flash",
        "models/gemini-2.5-flash-lite"
    ]

    for m in models:
        try:
            model = genai.GenerativeModel(m)
            _ = model.count_tokens("health_check")
            log_event(f"✅ Gemini model ready: {m}")
            print(f"✅ Gemini model ready: {m}")
            return model
        except Exception as e:
            log_event(f"⚠️ Model failed: {m} → {str(e)[:80]}")
            continue

    raise RuntimeError("❌ No working Gemini model found. Check API key or network connection.")


# Cache global model
GEMINI_MODEL = get_gemini_model()



def call_gemini_fallback(record):
    """
    Calls Gemini to predict a label and generate farming advice.
    Also saves new data to CSV and triggers model retraining.
    """
    prompt = f"""
You are an expert agronomist. Analyze the following data and respond ONLY in valid JSON:
Crop: {record['crop_name']}
Location: {record['location']}
Weather: {record.get('weather', 'unknown')}
Soil Type: {record['soiltype']}
Temperature: {record['temperature']}°C
Humidity: {record['humidity']}%
Rainfall: {record['rainfall']} mm

Respond in JSON exactly like this:
{{
  "label": "healthy" | "moderate" | "stress",
  "suggestion": "short practical farming advice"
}}
"""

    try:
        response = GEMINI_MODEL.generate_content(prompt)
        text = response.text.strip()

        # Extract valid JSON from model response
        if "{" in text and "}" in text:
            text = text[text.index("{"): text.rindex("}") + 1]

        data = json.loads(text)
        label = data.get("label", "moderate")
        suggestion = data.get(
            "suggestion",
            "Maintain regular monitoring and adjust irrigation as needed."
        )

    except Exception as e:
        log_event(f"⚠️ Gemini parse or response error: {e}")
        label = "moderate"
        suggestion = "Unable to interpret Gemini response. Monitor the crop manually."

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

    try:
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
            log_event(f"📦 Added new Gemini record: {new_entry['location']} ({label})")
            retrain_model()
            log_event("🔁 Model retrained after Gemini fallback.")
        else:
            log_event(f"⚠️ Duplicate record for {new_entry['location']} skipped.")
    except Exception as e:
        log_event(f"❌ Failed to save or retrain after Gemini fallback: {e}")

    return label, suggestion
