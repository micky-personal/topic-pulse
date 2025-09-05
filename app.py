from flask import Flask, jsonify, request
import news_scraper
import social_media_scraper
import sentiment_analyzer
# import firebase_storage

app = Flask(__name__)


@app.route('/topic_analysis', methods=['POST'])
def get_topic_analysis():
    """
    API endpoint to perform analysis for a given topic.
    """
    data = request.get_json()
    topic = data.get('topic')
    sub_topic = data.get('sub_topic')
    # caller_id = data.get('caller_id')

    if not topic:
        return jsonify({"error": "Missing 'topic' in request body"}), 400
    if not sub_topic:
        return jsonify({"error": "Missing 'sub-topic' in request body"}), 400

    # Step 1: Scrape data from multiple sources
    news_headlines = news_scraper.scrape_news(topic, sub_topic)
    social_posts = social_media_scraper.scrape_social_media(topic, sub_topic)
    all_text = news_headlines + social_posts

    # Step 2: Perform sentiment analysis
    signal, highlights = sentiment_analyzer.analyze_sentiment(all_text)

    # Prepare the response with references
    response_data = {
        "topic": topic,
        "signal": signal,
        "highlights": highlights,
        "references": {
            "news_source": "Moneycontrol",
            "social_media_source": "Mock Social Media"
        }
    }

    # Step 3: Store the highlights and references in Firestore
    # if caller_id:
    #     firebase_storage.store_sentiment_data(response_data, caller_id)

    return jsonify(response_data)


if __name__ == '__main__':
    app.run()