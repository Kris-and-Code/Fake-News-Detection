class MultiModalAnalyzer:
    def __init__(self):
        # Weights for different modalities
        self.text_weight = 0.6
        self.image_weight = 0.4

    def combine_scores(self, text_score, image_scores):
        """
        Combine text and image analysis scores into a final prediction
        Returns a score between 0 (likely true) and 1 (likely fake)
        """
        if not image_scores:
            # If no images provided, return text score
            return text_score
        
        # Calculate average image score
        avg_image_score = sum(image_scores) / len(image_scores)
        
        # Combine scores with weights
        final_score = (
            self.text_weight * text_score +
            self.image_weight * avg_image_score
        )
        
        return final_score

    def get_confidence_level(self, score):
        """
        Convert numerical score to confidence level
        """
        if score < 0.3:
            return "High confidence - Likely True"
        elif score < 0.5:
            return "Moderate confidence - Possibly True"
        elif score < 0.7:
            return "Moderate confidence - Possibly False"
        else:
            return "High confidence - Likely False"

    def get_explanation(self, text_score, image_scores, final_score):
        """
        Generate explanation for the analysis results
        """
        explanation = []
        
        # Text analysis explanation
        if text_score > 0.7:
            explanation.append("The text content shows strong indicators of potential misinformation.")
        elif text_score > 0.5:
            explanation.append("The text content shows some suspicious patterns.")
        else:
            explanation.append("The text content appears to be reliable.")
        
        # Image analysis explanation
        if image_scores:
            avg_image_score = sum(image_scores) / len(image_scores)
            if avg_image_score > 0.7:
                explanation.append("The associated images show signs of manipulation.")
            elif avg_image_score > 0.5:
                explanation.append("Some images show suspicious patterns.")
            else:
                explanation.append("The images appear to be authentic.")
        
        # Final verdict
        explanation.append(f"\nOverall confidence: {self.get_confidence_level(final_score)}")
        
        return " ".join(explanation) 