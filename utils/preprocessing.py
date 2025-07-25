from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer

lemmatizer = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()

def tokenize_and_lemmatize(text):
    words = tokenizer.tokenize(text)
    return [lemmatizer.lemmatize(w.lower()) for w in words]

def bag_of_words(words, vocabulary):
    return [1 if word in words else 0 for word in vocabulary]

