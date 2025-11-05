from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
from infer import predict_and_suggest

app = Flask(__name__)
CORS(app)  

def to_lowercase(data):
    """Convert all string values (and keys) in the dict to lowercase."""
    lowered = {}
    for key, value in data.items():
        if isinstance(value, str):
            lowered[key.lower()] = value.lower()
        else:
            lowered[key.lower()] = value
    return lowered


@app.route("/suggest", methods=["POST"])
def suggest():
    try:
        
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        data = to_lowercase(data)

        required_fields = ["crop", "location", "soil"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

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

        label, suggestion = predict_and_suggest(record)

        return jsonify({
            "prediction": str(label),
            "suggestion": suggestion,
            "processed_data": record
        })

    except Exception as e:
        print("Error during /suggest processing:")
        print(traceback.format_exc())  
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)