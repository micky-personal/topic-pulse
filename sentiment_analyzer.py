from textblob import TextBlob


def analyze_sentiment(text_data):
    """
    Performs sentiment analysis on a list of text strings.

    Args:
        text_data (list): A list of strings (e.g., news headlines, social media posts).

    Returns:
        tuple: A tuple containing (signal, highlights)
            - signal (str): 'POSITIVE', 'NEGATIVE', or 'NEUTRAL'
            - highlights (list): A list of key phrases/sentences.
    """
    if not text_data:
        return 'NEUTRAL', []

    total_polarity = 0
    highlights = []

    for text in text_data:
        analysis = TextBlob(text['text'])
        total_polarity += analysis.sentiment.polarity

        # Add a highlight for strong positive or negative sentences
        if analysis.sentiment.polarity > 0.4:
            highlights.append(f"Positive highlight: '{text}'")
        elif analysis.sentiment.polarity < -0.4:
            highlights.append(f"Negative highlight: '{text}'")

    average_polarity = total_polarity / len(text_data)

    if average_polarity > 0.15:
        signal = 'POSITIVE'
    elif average_polarity < -0.15:
        signal = 'NEGATIVE'
    else:
        signal = 'NEUTRAL'

    print(f"Sentiment analysis complete. Average polarity: {average_polarity:.2f}, Signal: {signal}")

    return signal, highlights