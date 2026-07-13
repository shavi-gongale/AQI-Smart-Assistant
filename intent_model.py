from nlp_engine import detect_intent, extract_city, extract_day_offset, split_questions

def parse_user_input(text):
    questions = split_questions(text)
    parsed = []
    for q in questions:
        intent = detect_intent(q)
        city = extract_city(q)
        day_offset = extract_day_offset(q)
        parsed.append({
            'text': q,
            'intent': intent,
            'city': city,
            'day_offset': day_offset
        })
    return parsed
