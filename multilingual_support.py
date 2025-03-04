from langdetect import detect
from googletrans import Translator

translator = Translator()

def detect_language(text: str) -> str:
    """Detect the language of the given text."""
    return detect(text)

def translate_to_english(text: str, src_lang: str) -> str:
    """Translate text to English."""
    if src_lang == 'en':
        return text
    translation = translator.translate(text, src='auto', dest='en')
    return translation.text

def translate_from_english(text: str, dest_lang: str) -> str:
    """Translate text from English to the destination language."""
    if dest_lang == 'en':
        return text
    translation = translator.translate(text, src='en', dest=dest_lang)
    return translation.text