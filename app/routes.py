from flask import Blueprint, request, jsonify
from app.models.text_analyzer import TextAnalyzer
from app.models.image_analyzer import ImageAnalyzer
from app.models.multimodal_analyzer import MultiModalAnalyzer

main = Blueprint('main', __name__)
text_analyzer = TextAnalyzer()
image_analyzer = ImageAnalyzer()
multimodal_analyzer = MultiModalAnalyzer()

@main.route('/api/analyze', methods=['POST'])
def analyze_news():
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text content provided'}), 400

        # Extract text and image URLs from request
        text_content = data['text']
        image_urls = data.get('images', [])

        # Analyze text
        text_score = text_analyzer.analyze(text_content)
        
        # Analyze images if provided (resilient per URL)
        image_scores = []
        if image_urls:
            for url in image_urls:
                try:
                    score = image_analyzer.analyze(url)
                except Exception as img_err:
                    # On error, append neutral score to avoid failing the whole request
                    print(f"Image analysis failed for {url}: {img_err}")
                    score = 0.5
                image_scores.append(score)
        
        # Combine results
        final_score = multimodal_analyzer.combine_scores(text_score, image_scores)
        
        return jsonify({
            'text_score': text_score,
            'image_scores': image_scores,
            'final_score': final_score,
            'is_fake': final_score > 0.5  # Threshold for fake news detection
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}) 