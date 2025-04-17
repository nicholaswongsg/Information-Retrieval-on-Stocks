import re
import pandas as pd
from spellchecker import SpellChecker
import os

# A dictionary of known stock-related synonyms or forced corrections
SYNONYMS = {
    # Apple variations
    "appl": "apple",
    "appin": "apple",
    "aplin": "apple",
    "appln": "apple",
    "aapl": "apple",
    
    # Stock tickers to company names
    "nvda": "nvidia",
    "amzn": "amazon",
    "msft": "microsoft",
    "goog": "google",
    "googl": "google",
    "meta": "meta",
    "fb": "meta",
    "tsla": "tesla",
    
    # Common misspellings
    "invester": "investor",
    "investers": "investors",
    "stonk": "stock",
    "stonks": "stocks",
    "stok": "stock",
    "stoks": "stocks",
    "portfoilo": "portfolio",
    "portflio": "portfolio",
    "divdend": "dividend",
    "divdends": "dividends",
}

def load_domain_dictionary(csv_file_path, text_column='ner_text_cleaned'):
    """Load domain-specific dictionary from the dataset"""
    try:
        df = pd.read_csv(csv_file_path)
        texts = df[text_column].dropna()
        
        all_words = []
        for line in texts:
            line_clean = re.sub(r'[^\w\s]', '', str(line).lower())
            words = line_clean.split()
            all_words.extend(words)
        
        return all_words
    except Exception as e:
        print(f"Error loading domain dictionary: {str(e)}")
        return []

def build_spellchecker(csv_file_path=None):
    """Build a spellchecker with domain-specific vocabulary"""
    # Increase distance to 3 for more flexible corrections
    spell = SpellChecker(distance=3)

    # Load domain words if CSV path is provided
    if csv_file_path and os.path.exists(csv_file_path):
        domain_words = load_domain_dictionary(csv_file_path)
        spell.word_frequency.load_words(domain_words)

    # Boost stock-related words; large frequency ensures they outrank common English words
    stock_names = [
        "apple", "tesla", "nvidia", "amazon", "google", "meta", "microsoft", 
        "berkshire", "hathaway", "jpmorgan", "netflix", "disney", "intel", "amd",
        "advanced", "micro", "devices", "ibm", "qualcomm", "adobe", "cisco", "paypal",
        "verizon", "att", "tmobile", "boeing", "ford", "gm", "general", "motors"
    ]
    
    for name in stock_names:
        spell.word_frequency.add(name, 1000000)  # Very high frequency to overshadow common words
    
    return spell

def correct_query(spell, user_query):
    """Correct a user query using the spellchecker"""
    query_clean = re.sub(r'[^\w\s]', '', user_query.lower())
    query_words = query_clean.split()
    
    corrected_words = []
    for w in query_words:
        # 1) Force known synonyms first
        if w in SYNONYMS:
            corrected_words.append(SYNONYMS[w])
        # 2) If recognized, keep as-is
        elif w in spell:
            corrected_words.append(w)
        else:
            # 3) Otherwise, use spell.correction
            correction = spell.correction(w)
            corrected_words.append(correction if correction else w)
    
    corrected_query = " ".join(corrected_words)
    return corrected_query

# Global spell checker instance
SPELL_CHECKER = None

def get_spell_checker(csv_file_path=None):
    """Get or initialize the global spell checker instance"""
    global SPELL_CHECKER
    if SPELL_CHECKER is None:
        SPELL_CHECKER = build_spellchecker(csv_file_path)
    return SPELL_CHECKER

def suggest_correction(user_query, csv_file_path=None):
    """Suggest a correction for the user query"""
    spell = get_spell_checker(csv_file_path)
    corrected_query = correct_query(spell, user_query)
    
    if corrected_query.lower() != user_query.lower():
        return corrected_query
    return None 