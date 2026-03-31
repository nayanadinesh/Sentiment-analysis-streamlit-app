import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# Slang dictionary
def replace_slang(text):
    slang_dict = {
        "lol": "laugh",
        "omg": "oh my god",
        "wtf": "what the hell",
        "idk": "i do not know",
        "smh": "disappointed",
        "tbh": "to be honest",
        "luv": "love"
    }

    words = text.split()
    words = [slang_dict.get(word, word) for word in words]
    return " ".join(words)

# Sarcasm indicator
def detect_sarcasm(text):
    sarcasm_words = [
        "yeah right",
        "sure",
        "totally",
        "great job",
        "wow",
        "nice",
        "amazing",
        "love that"
    ]

    for word in sarcasm_words:
        if word in text:
            return True
    return False

def clean_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtags symbol
    text = re.sub(r"#", "", text)

    # Replace slang
    text = replace_slang(text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]

    text = " ".join(words)

    return text