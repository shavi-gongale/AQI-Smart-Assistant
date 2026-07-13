"""
chatbot_engine.py
=================
Routes natural-language questions to the right handler (real-time AQI,
forecast, prediction, health advice, etc).

Now session-aware: the Streamlit page passes in the user's saved
features and the predicted AQI, so the chatbot's forecast / future-day
answers stay consistent with what the user just predicted on the
Prediction page.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from intent_model    import parse_user_input
from api             import get_real_aqi
from predict_aqi     import (predict_aqi, get_aqi_category,
                             get_aqi_risk, get_default_features)
from forecast        import forecast_next_7_days
from python_bridge   import ask_prolog_chat, is_prolog_available

PROLOG_OK = is_prolog_available()

AQI_ADVICE = {
    "outdoor": {
        "Good": "It's safe to go outside. Enjoy the fresh air!",
        "Moderate": "Mostly safe outside. Sensitive individuals should limit prolonged exposure.",
        "Unhealthy": "Avoid outdoor activity. Stay indoors if possible.",
        "Poor": "Stay indoors as much as possible. Avoid outdoor activities.",
        "Hazardous": "Do NOT go outside. This is a health emergency.",
    },
    "mask": {
        "Good": "No mask needed — air quality is good.",
        "Moderate": "Mask optional for sensitive individuals.",
        "Unhealthy": "Wearing a mask (N95/KN95) is recommended.",
        "Poor": "Mask is strongly recommended — use N95 or better.",
        "Hazardous": "Mask is essential. Use N95 and minimize exposure.",
    },
    "health": {
        "Good": "No health risks. Air quality is excellent.",
        "Moderate": "Minor irritation possible for very sensitive people.",
        "Unhealthy": "Breathing discomfort possible, especially for children and elderly.",
        "Poor": "Serious respiratory issues likely. Consult a doctor if you feel unwell.",
        "Hazardous": "Severe health danger. Seek medical attention if you experience symptoms.",
    },
    "exercise": {
        "Good": "Great day for outdoor exercise!",
        "Moderate": "Limit heavy outdoor exercise.",
        "Unhealthy": "Avoid outdoor exercise. Work out indoors.",
        "Poor": "Exercise only indoors with ventilation.",
        "Hazardous": "No outdoor activity. Rest indoors.",
    },
    "precaution": {
        "Good": "No special precautions needed.",
        "Moderate": "Stay hydrated, avoid prolonged outdoor exposure.",
        "Unhealthy": "Wear a mask, limit outdoor time, keep windows closed.",
        "Poor": "Stay indoors, use air purifier, keep windows closed tightly.",
        "Hazardous": "Avoid all outdoor exposure. Use air purifiers. Evacuate if possible.",
    },
    "children": {
        "Good": "Safe for children to play outside.",
        "Moderate": "Monitor children for any irritation.",
        "Unhealthy": "Keep children indoors.",
        "Poor": "Keep children strictly indoors.",
        "Hazardous": "Strict indoor stay for children — health emergency.",
    },
    "elder": {
        "Good": "Safe for elderly.",
        "Moderate": "Elderly should be cautious and limit exposure.",
        "Unhealthy": "Elderly should avoid outdoor exposure.",
        "Poor": "Elderly must stay indoors.",
        "Hazardous": "High risk for elderly. Stay indoors and seek medical help if needed.",
    },
    "pregnant": {
        "Good": "Safe conditions for pregnant women.",
        "Moderate": "Pregnant women should take precautions and limit outdoor time.",
        "Unhealthy": "Avoid outdoor exposure during pregnancy.",
        "Poor": "Pregnant women should stay indoors completely.",
        "Hazardous": "Extremely unsafe. Pregnant women must stay indoors and consult a doctor.",
    },
    "indoor": {
        "Good": "Indoor air is safe.",
        "Moderate": "Indoor air is slightly better than outside.",
        "Unhealthy": "Use an air purifier indoors.",
        "Poor": "Keep windows tightly closed. Use air purifier.",
        "Hazardous": "Stay indoors with air purifier running continuously.",
    },
    "symptoms": {
        "Good": "No pollution-related symptoms expected.",
        "Moderate": "Mild eye or throat irritation possible for sensitive people.",
        "Unhealthy": "Coughing, shortness of breath, and eye irritation are common.",
        "Poor": "Serious symptoms: wheezing, chest pain, difficulty breathing.",
        "Hazardous": "Severe symptoms possible. Seek medical attention immediately.",
    },
}

GENERAL_RESPONSES = {
    "cause":       "AQI increases due to vehicle emissions, industrial pollution, "
                   "burning of waste/crop residue, construction dust, and weather "
                   "conditions like low wind speed and temperature inversion.",
    "improve":     "To improve air quality: plant trees, use public transport, "
                   "reduce burning of waste, switch to clean energy, and support "
                   "environmental regulations.",
    "general_aqi": "AQI (Air Quality Index) measures how clean or polluted the air "
                   "is. Scale: 0-50 Good, 51-100 Moderate, 101-200 Unhealthy, "
                   "201-300 Poor, 300+ Hazardous.",
}


def _format_forecast_table(forecast):
    days = ["Today+1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"]
    lines = ["**7-Day AQI Forecast:**\n",
             "| Day | Predicted AQI | Category |",
             "|-----|--------------|----------|"]
    for day, aqi in zip(days, forecast):
        lines.append(f"| {day} | {aqi} | {get_aqi_category(aqi)} |")
    return "\n".join(lines)


def _handle_single_query(parsed,
                         context_city=None,
                         context_aqi=None,
                         saved_features=None,
                         saved_predicted_aqi=None):
    """One question, one answer."""
    intent     = parsed['intent']
    city       = parsed.get('city') or context_city
    day_offset = parsed.get('day_offset')

    # Use the user's saved features when available, otherwise defaults.
    features = list(saved_features) if saved_features else get_default_features()
    parts    = []
    active_aqi = None

    # ── REAL-TIME ────────────────────────────────────────────────────────────
    if intent == 'realtime':
        target_city = city or "delhi"
        aqi, city_name = get_real_aqi(target_city)
        if aqi is not None:
            cat  = get_aqi_category(aqi)
            risk = get_aqi_risk(aqi)
            parts.append(f"**Real-time AQI for {city_name}:** {aqi} ({cat}) — Risk: {risk}")
            active_aqi = aqi
        else:
            parts.append(f"Could not fetch real-time AQI for **{target_city}**. "
                         "Try a major city like Delhi, Mumbai, or London.")

    # ── 7-DAY FORECAST ───────────────────────────────────────────────────────
    elif intent == 'forecast':
        anchor = None
        if city:
            real_aqi, city_name = get_real_aqi(city)
            if real_aqi:
                parts.append(f"**Current AQI for {city_name}:** {real_aqi} ({get_aqi_category(real_aqi)})\n")
                anchor = real_aqi
        if anchor is None and saved_predicted_aqi is not None:
            anchor = float(saved_predicted_aqi)
        forecast = forecast_next_7_days(features, base_aqi=anchor)
        active_aqi = forecast[0]
        parts.append(_format_forecast_table(forecast))

    # ── FUTURE DAY ───────────────────────────────────────────────────────────
    elif intent == 'future_day':
        if day_offset:
            anchor = None
            if city:
                real_aqi, _ = get_real_aqi(city)
                if real_aqi:
                    anchor = real_aqi
            if anchor is None and saved_predicted_aqi is not None:
                anchor = float(saved_predicted_aqi)

            forecast = forecast_next_7_days(features, base_aqi=anchor)
            if 1 <= day_offset <= 7:
                pred_aqi = forecast[day_offset - 1]
                cat      = get_aqi_category(pred_aqi)
                parts.append(f"**AQI Forecast for Day +{day_offset}:** {pred_aqi} ({cat})")
                adv = AQI_ADVICE["precaution"].get(cat, "")
                if adv:
                    parts.append(f"💡 {adv}")
                active_aqi = pred_aqi
            else:
                parts.append("I can only forecast up to 7 days ahead.")
        else:
            parts.append("Please specify how many days ahead (e.g. 'after 3 days' or 'tomorrow').")

    # ── ML PREDICTION ────────────────────────────────────────────────────────
    elif intent == 'prediction':
        pred = saved_predicted_aqi if saved_predicted_aqi is not None \
                                   else predict_aqi(features)
        if pred is not None:
            cat  = get_aqi_category(pred)
            risk = get_aqi_risk(pred)
            parts.append(f"**ML-Predicted AQI:** {round(pred, 1)} ({cat}) — Risk: {risk}")
            parts.append("*(Based on the parameters you currently have set.)*")
            active_aqi = pred
        else:
            parts.append("Prediction failed. Please check that the model files are available.")

    # ── HEALTH / ADVICE INTENTS ──────────────────────────────────────────────
    elif intent in AQI_ADVICE:
        active_aqi = context_aqi
        if city and active_aqi is None:
            aqi_val, _ = get_real_aqi(city)
            if aqi_val:
                active_aqi = aqi_val
        if active_aqi is None and saved_predicted_aqi is not None:
            active_aqi = float(saved_predicted_aqi)
        if active_aqi is None:
            active_aqi = predict_aqi(features) or 100

        cat    = get_aqi_category(active_aqi)
        advice = AQI_ADVICE[intent].get(cat, "")

        if PROLOG_OK:
            prolog_resp = ask_prolog_chat(active_aqi, parsed['text'])
            if prolog_resp:
                parts.append(f"🤖 {prolog_resp}")
            elif advice:
                parts.append(f"💬 {advice}")
        elif advice:
            parts.append(f"💬 {advice}")

        if not parts:
            parts.append(f"Current AQI context: {round(active_aqi)} ({cat}). {advice}")

    # ── GENERAL FACTS ────────────────────────────────────────────────────────
    elif intent in GENERAL_RESPONSES:
        parts.append(f"💬 {GENERAL_RESPONSES[intent]}")

    # ── FALLBACK ─────────────────────────────────────────────────────────────
    else:
        active_aqi = context_aqi
        if active_aqi is None and saved_predicted_aqi is not None:
            active_aqi = float(saved_predicted_aqi)
        if active_aqi is None:
            active_aqi = predict_aqi(features) or 100
        cat = get_aqi_category(active_aqi)
        if PROLOG_OK:
            prolog_resp = ask_prolog_chat(active_aqi, parsed['text'])
            if prolog_resp:
                parts.append(f"🤖 {prolog_resp}")
        if not parts:
            parts.append(f"I understand you're asking about air quality. "
                         f"Current estimated AQI is {round(active_aqi)} ({cat}).")
            parts.append("You can ask me about: current AQI, 7-day forecast, "
                         "health advice, masks, outdoor safety, and more.")

    return "\n\n".join(parts), city, (context_aqi or active_aqi)


def get_response(user_input,
                 context_city=None,
                 context_aqi=None,
                 saved_features=None,
                 saved_predicted_aqi=None):
    """
    Public entry point.

    Pass `saved_features` and `saved_predicted_aqi` from
    `st.session_state` so the chatbot stays consistent with the
    Prediction / Forecast pages.
    """
    if not user_input or not user_input.strip():
        return "Please ask me something about air quality!", context_city, context_aqi

    parsed_list  = parse_user_input(user_input)
    responses    = []
    current_city = context_city
    current_aqi  = context_aqi

    for i, parsed in enumerate(parsed_list):
        resp, city, aqi = _handle_single_query(
            parsed, current_city, current_aqi,
            saved_features=saved_features,
            saved_predicted_aqi=saved_predicted_aqi,
        )
        if city: current_city = city
        if aqi:  current_aqi  = aqi
        if len(parsed_list) > 1:
            responses.append(f"**Q{i+1}: {parsed['text']}**\n{resp}")
        else:
            responses.append(resp)

    final = "\n\n---\n\n".join(responses) if len(responses) > 1 else responses[0]
    return final, current_city, current_aqi
