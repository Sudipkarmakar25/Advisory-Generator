import joblib
import numpy as np
import pandas as pd
from logger_utils import log_event

MODEL_OUT = "crop_model.joblib"
ENC_OUT = "encoders.joblib"
DATA_FILE = "sample_data.csv"

clf = joblib.load(MODEL_OUT)
enc = joblib.load(ENC_OUT)

def safe_transform(enc, crop, location, weather, soil):
    try:
        return enc.transform([[crop, location, weather, soil]])
    except ValueError:
        known_crops, known_locations, known_weather, known_soil = enc.categories_
        crop = crop if crop in known_crops else known_crops[0]
        location = location if location in known_locations else known_locations[0]
        weather = weather if weather in known_weather else known_weather[0]
        soil = soil if soil in known_soil else known_soil[0]
        # log_event("Unknown category replaced safely.")
        return enc.transform([[crop, location, weather, soil]])
