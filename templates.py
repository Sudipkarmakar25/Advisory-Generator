import random
import textwrap
from cropInfo import CROP_LIFECYCLES  


# --- 1. Helper: Determine crop stage ---
def get_crop_stage(days_since_planting, total_lifecycle):
    if total_lifecycle == 0:
        return "unknown"
    ratio = days_since_planting / total_lifecycle
    if ratio <= 0.2:
        return "seedling"
    elif ratio <= 0.5:
        return "vegetative"
    elif ratio <= 0.8:
        return "flowering"
    else:
        return "maturity"


# --- 2. Main Suggestion Builder ---
def build_suggestion(pred_label, features):
    # --- Extract all features ---
    crop = features.get("crop_name", "your crop").lower()
    lifecycle = features.get("lifecycle_days", 0)
    days_since = features.get("days_since_planting", 0)
    temp = features.get("temperature", 0)
    hum = features.get("humidity", 0)
    rain = features.get("rainfall", 0)
    soil = features.get("soiltype", "soil")
    location = features.get("location", "your area").capitalize()
    weather = features.get("weather", "normal")
    farmer = features.get("farmer_name", "Farmer")

    # --- Automatically fetch lifecycle if missing ---
    if lifecycle == 0:
        lifecycle = CROP_LIFECYCLES.get(crop, 100)  # default to 100 days if unknown

    # --- Determine crop stage ---
    stage = get_crop_stage(days_since, lifecycle)

    # --- Define fertilizer and weed control by stage ---
    fertilizer_by_stage = {
        "seedling": "This stage needs Phosphorus (P) for root growth. Use a starter fertilizer (e.g., high-P, low-N) suitable for {soil} soil.",
        "vegetative": "The crop is in active growth. Apply a high-Nitrogen (N) fertilizer to support leaf and stem development.",
        "flowering": "Shift nutrient focus. The crop needs more Phosphorus (P) and Potassium (K) to support flower and fruit set. Reduce Nitrogen.",
        "maturity": "Nutrient requirements are low. No major fertilizer application is needed. Focus on harvest preparation.",
        "unknown": "Apply a balanced fertilizer (e.g., NPK 10-10-10) suitable for {soil} soil and your {crop}."
    }

    weed_control_by_stage = {
        "seedling": "This is a CRITICAL period. Young plants cannot compete with weeds. Use manual weeding or a recommended pre-emergence herbicide for {crop}.",
        "vegetative": "Weeds are still a major threat. Monitor and apply a post-emergence herbicide if needed, before the crop canopy closes.",
        "flowering": "Weed competition is less critical, but remove large weeds that block sunlight or moisture.",
        "maturity": "Weed control is not usually needed now; focus on harvest preparation.",
        "unknown": "Maintain regular weed monitoring and control practices for your {crop}."
    }

    fert_advice = fertilizer_by_stage.get(stage, fertilizer_by_stage["unknown"]).format(soil=soil, crop=crop)
    weed_advice = weed_control_by_stage.get(stage, weed_control_by_stage["unknown"]).format(crop=crop)

    # --- Define messages based on model prediction label ---
    messages = {
        "irrigation_needed": [
            f"Hello {farmer}, your {crop} field in {location} shows low moisture under {weather} weather. "
            f"At {temp}°C and {hum}% humidity, irrigation is required for healthy growth."
        ],
        "reduce_irrigation": [
            f"Hello {farmer}, your {crop} field in {location} has sufficient moisture after {rain}mm rainfall. "
            f"You can reduce irrigation temporarily."
        ],
        "fertilizer": [
            f"Hello {farmer}, your {crop} crop is in the {stage} stage. {fert_advice} "
            f"Conditions ({temp}°C, {hum}% humidity) are suitable for fertilizer application."
        ],
        "pest_monitor": [
            f"Hello {farmer}, warm and humid conditions ({temp}°C, {hum}% humidity) can attract pests in your {crop} field. "
            f"Inspect plants daily and use eco-friendly pest control methods."
        ],
        "weed_control": [
            f"Hello {farmer}, your {crop} in {location} is at the {stage} stage. {weed_advice}"
        ],
        "normal_monitor": [
            f"Hello {farmer}, conditions are stable for your {crop} in {location}. "
            f"Continue regular monitoring and irrigation as needed."
        ]
    }

    # --- Pick one message and add hints ---
    base_message = random.choice(messages.get(pred_label, messages["normal_monitor"]))

    stage_hints = {
        "seedling": "Avoid overwatering and protect young plants.",
        "vegetative": "Ensure proper irrigation for vigorous growth.",
        "flowering": "Maintain consistent moisture and check for pollination success.",
        "maturity": "Reduce irrigation and prepare for harvest."
    }

    hint = ""
    if pred_label not in ["fertilizer", "weed_control"]:
        hint = stage_hints.get(stage, "")

    paragraph = f"{base_message} {hint}"

    # --- Format message neatly ---
    words = paragraph.split()
    if len(words) > 55:
        words = words[:50]
    elif len(words) < 30:
        words += ["Keep", "observing", "the", "field", "daily", "for", "any", "change."]

    return textwrap.fill(" ".join(words), width=80)


# --- Example Usage ---
if __name__ == "__main__":
    features_1 = {
        "crop_name": "Pineapple", "days_since_planting": 300,
        "temperature": 30, "humidity": 70, "rainfall": 5,
        "soiltype": "sandy loam", "location": "Plot X",
        "weather": "sunny", "farmer_name": "Ravi"
    }
    print("--- Irrigation Needed (Pineapple) ---")
    print(build_suggestion("irrigation_needed", features_1))
    print("\n")

    features_2 = {
        "crop_name": "Brinjal", "days_since_planting": 50,
        "temperature": 29, "humidity": 65, "rainfall": 3,
        "soiltype": "clay", "location": "Plot Y",
        "weather": "cloudy", "farmer_name": "Priya"
    }
    print("--- Fertilizer (Brinjal) ---")
    print(build_suggestion("fertilizer", features_2))
