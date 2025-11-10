import numpy as np
import textwrap
from templates import build_suggestion
from model_utils import clf, enc, safe_transform
from gemini_utils import call_gemini_fallback
from logger_utils import log_event

def predict_and_suggest(record):
    crop = record["crop_name"]
    location = record["location"]
    weather = record.get("weather", "normal")
    soil = record.get("soiltype", "loamy")
    temperature = record["temperature"]
    humidity = record["humidity"]
    rainfall = record["rainfall"]

    try:
        X_cat = safe_transform(enc, crop, location, weather, soil)
        X_num = np.array([[temperature, humidity, rainfall]])
        X = np.hstack([X_cat, X_num])
        probs = clf.predict_proba(X)[0]
        confidence = max(probs)
        pred_label = clf.classes_[np.argmax(probs)]
        # log_event(f"Prediction: {pred_label} (confidence={confidence:.2f})")
        print(confidence)
        if confidence < 0.5:
            # log_event("Low confidence → Gemini fallback")
            return call_gemini_fallback(record)

        stage_suggestion = build_suggestion(pred_label, record)
        irrigation_msg = (
            "No rainfall — irrigate lightly." if rainfall == 0
            else f"Low rainfall ({rainfall}mm) — maintain watering."
            if rainfall < 30
            else f"Sufficient rainfall ({rainfall}mm) — reduce irrigation."
        )
        temp_msg = (
            f"High temperature ({temperature}°C) — use mulch."
            if temperature > 35
            else f"Cool temperature ({temperature}°C) — ensure sunlight."
            if temperature < 20
            else f"Temperature ({temperature}°C) is suitable."
        )
        hum_msg = (
            f"Low humidity ({humidity}%) — monitor dryness."
            if humidity < 40
            else f"High humidity ({humidity}%) — check for fungus."
            if humidity > 80
            else f"Humidity ({humidity}%) is ideal."
        )

        weather_summary = f"{irrigation_msg} {temp_msg} {hum_msg}"
        farmer = record.get("farmer_name", "Farmer").capitalize()
        crop_cap = crop.capitalize()
        location_cap = location.capitalize()

        suggestion = (
            f"Hello {farmer}, here is today's advisory for your {crop_cap} field in {location_cap}.\n\n"
            f"Weather Summary:\n{weather_summary}\n\n"
            f"Crop Advisory:\n{stage_suggestion}"
        )

        return pred_label, textwrap.fill(suggestion, width=90)

    except Exception as e:
        log_event(f"Inference error: {e}")
        return call_gemini_fallback(record)


if __name__ == "__main__":
    sample = {
        "crop_name": "rice",
        "location": "puruliawestbengal",
        "weather": "humid",
        "soiltype": "alluvial",
        "temperature": 1.5,
        "humidity": 22,
        "rainfall": 10,
        "farmer_name": "Sudip"
    }
    label, sug = predict_and_suggest(sample)
    print("Predicted label:", label)
    print("\nSuggestion:\n")
    print(sug)
