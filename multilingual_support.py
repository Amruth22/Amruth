from langdetect import detect
from langchain_openai import ChatOpenAI

# Initialize the GPT-4o model
llm = ChatOpenAI(model="gpt-4o", temperature=0)

def detect_language(text: str) -> str:
    """Detect the language of the given text."""
    return detect(text)

def translate_with_gpt(text: str, src_lang: str, dest_lang: str) -> str:
    """Translate text using GPT-4o."""
    if src_lang == dest_lang:
        return text
    
    # Create a translation prompt
    prompt = f"Translate the following text from {src_lang} to {dest_lang}:\n\n{text}"
    
    # Use GPT-4o to translate
    response = llm.invoke(prompt)
    return response.content.strip()