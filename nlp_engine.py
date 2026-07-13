import re

CITY_LIST = [
    "delhi", "mumbai", "kolkata", "chennai", "bangalore", "bengaluru",
    "hyderabad", "pune", "ahmedabad", "jaipur", "lucknow", "kanpur",
    "nagpur", "indore", "bhopal", "patna", "agra", "varanasi",
    "surat", "chandigarh", "guwahati", "raipur", "ranchi", "shimla",
    "visakhapatnam", "london", "paris", "beijing", "tokyo", "new york",
    "los angeles", "dubai", "singapore", "sydney"
]

def extract_city(text):
    text_lower = text.lower()
    for city in CITY_LIST:
        if city in text_lower:
            return city
    return None

def extract_day_offset(text):
    text_lower = text.lower()
    patterns = [
        (r'after\s+(\d+)\s+days?', lambda m: int(m.group(1))),
        (r'in\s+(\d+)\s+days?', lambda m: int(m.group(1))),
        (r'(\d+)\s+days?\s+later', lambda m: int(m.group(1))),
        (r'day\s+(\d+)', lambda m: int(m.group(1))),
        (r'tomorrow', lambda m: 1),
        (r'day after tomorrow', lambda m: 2),
        (r'next week', lambda m: 7),
    ]
    for pattern, extractor in patterns:
        match = re.search(pattern, text_lower)
        if match:
            return extractor(match)
    return None

def detect_intent(text):
    text_lower = text.lower()

    if any(w in text_lower for w in ['forecast', '7 day', '7-day', 'week', 'next few days']):
        return 'forecast'
    if any(w in text_lower for w in ['after', 'in 2 days', 'in 3 days', 'tomorrow', 'next day']):
        if extract_day_offset(text_lower):
            return 'future_day'
    if any(w in text_lower for w in ['predict', 'prediction', 'calculate', 'estimate']):
        return 'prediction'
    if any(w in text_lower for w in ['mask', 'wear mask', 'face cover']):
        return 'mask'
    if any(w in text_lower for w in ['exercise', 'gym', 'outdoor activity']):
        return 'exercise'
    if any(w in text_lower for w in ['outside', 'go out', 'step out']):
        return 'outdoor'
    if any(w in text_lower for w in ['child', 'kids', 'baby']):
        return 'children'
    if any(w in text_lower for w in ['elder', 'old people', 'senior']):
        return 'elder'
    if any(w in text_lower for w in ['pregnant', 'pregnancy']):
        return 'pregnant'
    if any(w in text_lower for w in ['symptom', 'cough', 'breathing', 'sneeze']):
        return 'symptoms'
    if any(w in text_lower for w in ['precaution', 'care', 'protect']):
        return 'precaution'
    if any(w in text_lower for w in ['cause', 'why', 'reason']):
        return 'cause'
    if any(w in text_lower for w in ['improve', 'solution', 'reduce', 'fix']):
        return 'improve'
    if any(w in text_lower for w in ['indoor', 'inside']):
        return 'indoor'
    if any(w in text_lower for w in ['health', 'effect', 'impact', 'risk']):
        return 'health'
    if any(w in text_lower for w in ['real', 'current', 'now', 'today', 'live', 'actual', 'aqi of', 'aqi in']):
        return 'realtime'
    if any(w in text_lower for w in ['aqi', 'air quality', 'pollution', 'polluted']):
        return 'general_aqi'

    return 'general'

def split_questions(text):
    parts = re.split(r'[?]\s*(?=[A-Z]|and |also |what|how|is |will |can )|(?:\band\b|\balso\b)(?=\s+(?:what|how|tell|is|will|can))', text, flags=re.IGNORECASE)
    questions = [p.strip().rstrip('?').strip() for p in parts if len(p.strip()) > 5]
    return questions if len(questions) > 1 else [text]
