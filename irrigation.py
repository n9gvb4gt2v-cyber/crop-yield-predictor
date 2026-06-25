CROP_WATER_DEMAND_MM = {
    "Rice": (900, 1200), "Wheat": (450, 650), "Maize": (500, 800),
    "Bajra": (300, 500), "Jowar": (300, 500), "Ragi": (350, 500),
    "Barley": (450, 600), "Small millets": (300, 500), "Other Cereals": (400, 600),
    "Gram": (300, 400), "Arhar/Tur": (500, 700), "Moong(Green Gram)": (300, 450),
    "Urad": (300, 450), "Masoor": (250, 350), "Cowpea(Lobia)": (300, 500),
    "Horse-gram": (250, 400), "Khesari": (250, 350), "Moth": (200, 350),
    "Peas & beans (Pulses)": (350, 500), "Other Kharif pulses": (300, 500),
    "Other  Rabi pulses": (300, 450), "Other Summer Pulses": (350, 500),
    "Groundnut": (500, 700), "Rapeseed &Mustard": (300, 450), "Sunflower": (600, 900),
    "Soyabean": (450, 700), "Sesamum": (300, 450), "Linseed": (300, 450),
    "Safflower": (600, 900), "Castor seed": (400, 600), "Niger seed": (350, 500),
    "Guar seed": (250, 400), "Oilseeds total": (400, 650), "other oilseeds": (400, 650),
    "Sugarcane": (1500, 2500), "Cotton(lint)": (700, 1300), "Jute": (500, 800),
    "Sannhamp": (500, 700), "Mesta": (500, 700), "Tobacco": (400, 600),
    "Arecanut": (1500, 2000), "Black pepper": (1200, 2000), "Cardamom": (1500, 2500),
    "Coriander": (300, 500), "Dry chillies": (600, 900), "Ginger": (1500, 1800),
    "Turmeric": (1500, 2000), "Garlic": (350, 500), "Banana": (1200, 2200),
    "Coconut ": (1500, 2000), "Cashewnut": (900, 1500), "Onion": (350, 550),
    "Potato": (500, 700), "Sweet potato": (500, 750), "Tapioca": (1000, 1500),
}

SEASON_FACTOR = {
    "Kharif": 1.0, "Rabi": 0.85, "Summer": 1.25,
    "Autumn": 0.90, "Winter": 0.80, "Whole Year": 1.40,
}

def get_irrigation_suggestion(crop, season, annual_rainfall_mm):
    demand_range = CROP_WATER_DEMAND_MM.get(crop, (400, 700))
    base_demand  = (demand_range[0] + demand_range[1]) / 2
    crop_demand  = base_demand * SEASON_FACTOR.get(season, 1.0)
    effective_rain = annual_rainfall_mm * 0.70
    irrigation_needed = max(0.0, crop_demand - effective_rain)

    if irrigation_needed == 0:
        status  = "sufficient"
        message = f"Rainfall is sufficient for {crop} in {season} season. No additional irrigation needed."
    elif irrigation_needed < 200:
        status  = "low_deficit"
        message = f"{crop} needs a small supplement of {irrigation_needed:.0f}mm. Light irrigation at key growth stages is recommended."
    elif irrigation_needed < 600:
        status  = "moderate_deficit"
        message = f"{crop} has a moderate water deficit of {irrigation_needed:.0f}mm. Regular irrigation scheduling is important."
    else:
        status  = "high_deficit"
        message = f"{crop} has a high water deficit of {irrigation_needed:.0f}mm. Intensive irrigation is critical."

    return {
        "crop": crop, "season": season,
        "annual_rainfall_mm": round(annual_rainfall_mm, 1),
        "effective_rainfall_mm": round(effective_rain, 1),
        "crop_water_demand_mm": round(crop_demand, 1),
        "irrigation_needed_mm": round(irrigation_needed, 1),
        "status": status, "message": message,
        "fao_source": "FAO Irrigation and Drainage Paper No. 56",
    }