# Fake News Detection with Multi-Modal Analysis

This project implements a sophisticated fake news detection system that combines multiple modalities of analysis to identify potentially misleading or false information. The system analyzes both textual content and associated images to provide a more robust assessment of news authenticity.

## Features

- **Text Analysis**: Natural Language Processing (NLP) to analyze article content, headlines, and metadata
- **Image Analysis**: Computer vision techniques to detect manipulated or suspicious images
- **Multi-Modal Integration**: Combines text and image features for comprehensive analysis
- **Web API**: RESTful API endpoint for easy integration
- **Modern Web Interface**: User-friendly interface for news verification

## Project Structure

```
├── app/
│   ├── __init__.py
│   ├── models/
│   │   ├── text_analyzer.py
│   │   ├── image_analyzer.py
│   │   └── multimodal_analyzer.py
│   ├── static/
│   │   ├── css/
│   │   └── js/
│   └── templates/
├── data/
│   └── models/
├── tests/
├── requirements.txt
└── run.py
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python run.py
```

## API Endpoints

- `POST /api/analyze`: Submit news article for analysis
  - Accepts JSON with text content and image URLs
  - Returns analysis results with confidence scores

## Models

### Text Analysis
- Uses BERT-based model for text classification
- Analyzes linguistic patterns and content credibility

### Image Analysis
- Implements image manipulation detection
- Uses deep learning models for visual content analysis

### Multi-Modal Integration
- Combines text and image features
- Weighted scoring system for final prediction

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 