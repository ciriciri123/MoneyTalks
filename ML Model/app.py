from flask import Flask, render_template, request, jsonify
from inference import predict_currency

app = Flask(__name__)

# Config
app.config['JSON_SORT_KEYS'] = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test')
def test_page():
    return render_template('test.html')

@app.route('/api/detect', methods=['POST'])
def detect():
    if 'frame' not in request.files:
        return jsonify({"error": "No frame received"}), 400
    
    file = request.files['frame']
    image_bytes = file.read()
    
    if not image_bytes:
        return jsonify({"error": "Empty frame"}), 400
        
    # Process
    result = predict_currency(image_bytes)
    
    if "error" in result:
        return jsonify({"error": result["error"]}), 500
        
    label = result['label']
    confidence = result['confidence']
    
    # As per PRD, > 0.75 confidence threshold is required to return label
    # Otherwise, it asks to move closer.
    
    # Mapping label string from dataset to human friendly name (idr_1000 -> 1000 Rupiah)
    amounts = {
        'idr_1000': 'Seribu Rupiah',
        'idr_2000': 'Dua Ribu Rupiah',
        'idr_5000': 'Lima Ribu Rupiah',
        'idr_10000': 'Sepuluh Ribu Rupiah',
        'idr_20000': 'Dua Puluh Ribu Rupiah',
        'idr_50000': 'Lima Puluh Ribu Rupiah',
        'idr_100000': 'Seratus Ribu Rupiah'
    }
    
    friendly_label = amounts.get(label, "Uang tidak dikenali")
    
    response_data = {
        "confidence": confidence,
        "raw_label": label
    }
    if "box" in result:
        response_data["box"] = result["box"]
    
    if confidence >= 0.75:
        response_data['message'] = friendly_label
        response_data['valid'] = True
    else:
        response_data['message'] = "Tolong dekatkan uang ke kamera"
        response_data['valid'] = False
        
    return jsonify(response_data)

if __name__ == '__main__':
    # Use Threaded to conform with ML operations and MediaDevices stream
    app.run(debug=True, threaded=True, host="0.0.0.0", port=5000)
