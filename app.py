from flask import Flask, jsonify, request
import trading_pulse.fetch_top_tickers
import trading_pulse.trading_prediction_model

app = Flask(__name__)

topic_subtopic_routing_map = {
    ('finance', 'topTradableTickers'): trading_pulse.fetch_top_tickers.fetch_top_tickers
}

ml_model_training_routing_map = {
    ('finance', 'topTradableTickers'): trading_pulse.trading_prediction_model.train_model
}

@app.route('/topic_analysis', methods=['POST'])
def get_topic_analysis():
    data = request.get_json()
    topic = data.get('topic')
    sub_topic = data.get('sub_topic')

    if not topic:
        return jsonify({"error": "Missing 'topic' in request body"}), 400
    if not sub_topic:
        return jsonify({"error": "Missing 'sub-topic' in request body"}), 400

    action_function = topic_subtopic_routing_map.get((topic, sub_topic))

    if action_function:
        result = action_function()
        if result is None:
            return jsonify({"error": f"Failed to get data for topic: {topic}"}), 500
        else:
            return jsonify({"topic": topic, "sub_topic": sub_topic, "result": result})

    else:
        data = {}

        response_data = {
            "topic": topic,
            "sub_topic": sub_topic,
            "data": data
        }

        return jsonify(response_data)

@app.route('/train_model', methods=['POST'])
def get_topic_analysis():
    data = request.get_json()
    topic = data.get('topic')
    sub_topic = data.get('sub_topic')
    training_data = data.get('training_data')

    if not topic:
        return jsonify({"error": "Missing 'topic' in request body"}), 400
    if not sub_topic:
        return jsonify({"error": "Missing 'sub-topic' in request body"}), 400

    action_function = ml_model_training_routing_map.get((topic, sub_topic))

    if action_function:
        action_function(training_data)
        return jsonify({"topic": topic, "sub_topic": sub_topic, "result": "Model trained successfully"}), 200

    else:
        return jsonify({"topic": topic, "sub_topic": sub_topic, "result": "No Model found"}), 200


if __name__ == '__main__':
    app.run()