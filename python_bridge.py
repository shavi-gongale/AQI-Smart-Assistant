import subprocess
import os

PROLOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aqi_smart_chatbot.pl")

def ask_prolog_chat(aqi, question):
    try:
        safe_question = question.replace('"', "'").replace('\\', '/')
        goal = f'chat({int(aqi)}, "{safe_question}", X), write(X), halt.'
        result = subprocess.run(
            ["swipl", "-q", "-s", PROLOG_FILE, "-g", goal],
            capture_output=True,
            text=True,
            timeout=10
        )
        answer = result.stdout.strip()
        if answer:
            return answer
        return None
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None

def is_prolog_available():
    try:
        result = subprocess.run(
            ["swipl", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False
