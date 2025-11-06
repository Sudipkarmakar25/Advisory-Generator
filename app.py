from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import os
from infer import predict_and_suggest

app = Flask(__name__)
CORS(app)  # Enable cross-origin requests (for React, etc.)

def to_lowercase(data):
    """Convert all string keys and string values in a dict to lowercase."""
    lowered = {}
    for key, value in data.items():
        lowered[key.lower()] = value.lower() if isinstance(value, str) else value
    return lowered


@app.route("/suggest", methods=["POST"])
def suggest():
    try:
        # Parse JSON input
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Normalize casing
        data = to_lowercase(data)

        # Validate essential fields
        required_fields = ["crop", "location", "soil"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Prepare record for model inference
        record = {
            "crop_name": data.get("crop"),
            "location": data.get("location"),
            "weather": data.get("weather", "unknown"),
            "soiltype": data.get("soil"),
            "temperature": float(data.get("temperature", 0)),
            "humidity": float(data.get("humidity", 0)),
            "rainfall": float(data.get("rainfall", 0)),
            "farmer_name": data.get("farmer_name", "Farmer")
        }

        # Run ML model + fallback logic
        label, suggestion = predict_and_suggest(record)

        # Return structured response
        return jsonify({
            "prediction": str(label),
            "suggestion": suggestion,
            "processed_data": record
        })

    except Exception as e:
        print("Error during /suggest processing:")
        print(traceback.format_exc())  # Log full error trace for debugging
        return jsonify({"error": str(e)}), 500


# ---- Dynamic port binding for Render ----
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Server running on port {port}")
    app.run(host="0.0.0.0", port=port)
