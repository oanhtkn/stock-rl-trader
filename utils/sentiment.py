from textblob import TextBlob

def get_sentiment_score(text: str) -> float:
    if not isinstance(text, str):
        return 0.0
    return TextBlob(text).sentiment.polarity
