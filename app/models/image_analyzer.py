import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import torch
from torchvision import transforms
try:
    # Newer torchvision API (>=0.13)
    from torchvision.models import resnet50, ResNet50_Weights
    _HAS_TV_WEIGHTS = True
except Exception:  # pragma: no cover
    # Older torchvision API fallback
    from torchvision.models import resnet50
    ResNet50_Weights = None
    _HAS_TV_WEIGHTS = False

class ImageAnalyzer:
    def __init__(self):
        # Initialize the model for image classification (handle multiple torchvision versions)
        if _HAS_TV_WEIGHTS and ResNet50_Weights is not None:
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.preprocess_norm_mean = ResNet50_Weights.DEFAULT.meta.get("mean", [0.485, 0.456, 0.406])
            self.preprocess_norm_std = ResNet50_Weights.DEFAULT.meta.get("std", [0.229, 0.224, 0.225])
        else:
            # Fallback for older versions where "weights" arg may not exist
            self.model = resnet50(pretrained=True)
            self.preprocess_norm_mean = [0.485, 0.456, 0.406]
            self.preprocess_norm_std = [0.229, 0.224, 0.225]

        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.preprocess_norm_mean,
                std=self.preprocess_norm_std
            )
        ])

    def analyze(self, image_url):
        """
        Analyze image for potential manipulation or suspicious content
        Returns a score between 0 (likely authentic) and 1 (likely manipulated)
        """
        try:
            # Download and process image with timeout and basic validation
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '')
            if 'image' not in content_type:
                raise ValueError(f"URL does not appear to be an image (Content-Type: {content_type})")

            img = Image.open(BytesIO(response.content)).convert('RGB')
            
            # Basic image analysis
            manipulation_score = self._detect_manipulation(img)
            content_score = self._analyze_content(img)
            
            # Combine scores (weighted average)
            final_score = 0.6 * manipulation_score + 0.4 * content_score
            
            return final_score
            
        except Exception as e:
            print(f"Error analyzing image: {str(e)}")
            return 0.5  # Neutral score on error

    def _detect_manipulation(self, img):
        """
        Detect potential image manipulation using various techniques
        """
        # Convert to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Basic manipulation detection
        # 1. Check for compression artifacts
        compression_score = self._check_compression_artifacts(img_cv)
        
        # 2. Check for edge inconsistencies
        edge_score = self._check_edge_inconsistencies(img_cv)
        
        # 3. Check for noise patterns
        noise_score = self._check_noise_patterns(img_cv)
        
        # Combine scores
        return (compression_score + edge_score + noise_score) / 3

    def _analyze_content(self, img):
        """
        Analyze image content for suspicious patterns
        """
        # Preprocess image for model
        img_tensor = self.transform(img).unsqueeze(0)
        if torch.cuda.is_available():
            img_tensor = img_tensor.cuda()

        # Get model predictions and compute entropy as a simple uncertainty heuristic
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)

        # Shannon entropy: higher entropy -> more uncertainty -> potentially suspicious
        # Normalize entropy to [0,1] by dividing by log(num_classes)
        num_classes = probs.shape[1]
        entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=1)
        entropy_norm = float(entropy.cpu().item() / (np.log(num_classes) + 1e-12))

        # Clamp to [0,1]
        entropy_norm = max(0.0, min(1.0, entropy_norm))
        return entropy_norm

    def _check_compression_artifacts(self, img):
        """
        Check for JPEG compression artifacts
        """
        # Simple heuristic: measure average absolute difference across 8-pixel grid boundaries
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        # Vertical boundaries every 8 columns
        vert_diff = []
        for x in range(8, w, 8):
            col_left = gray[:, x - 1].astype(np.float32)
            col_right = gray[:, x].astype(np.float32)
            vert_diff.append(np.mean(np.abs(col_right - col_left)))
        # Horizontal boundaries every 8 rows
        horiz_diff = []
        for y in range(8, h, 8):
            row_top = gray[y - 1, :].astype(np.float32)
            row_bottom = gray[y, :].astype(np.float32)
            horiz_diff.append(np.mean(np.abs(row_bottom - row_top)))

        diffs = vert_diff + horiz_diff
        if not diffs:
            return 0.5

        # Normalize by 255 to get 0-1 range and clamp
        score = float(np.mean(diffs) / 255.0)
        return max(0.0, min(1.0, score))

    def _check_edge_inconsistencies(self, img):
        """
        Check for edge inconsistencies that might indicate manipulation
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        # Compute edge density and edge cluster variance as a proxy for inconsistencies
        edge_density = np.mean(edges > 0)
        # Downscale to reduce noise for variance calculation
        small = cv2.resize(edges, (64, 64), interpolation=cv2.INTER_AREA)
        variance = float(np.var(small.astype(np.float32) / 255.0))

        # Combine with simple weighting; high density and high variance can indicate artifacts
        raw = 0.6 * edge_density + 0.4 * variance
        return max(0.0, min(1.0, raw))

    def _check_noise_patterns(self, img):
        """
        Check for unusual noise patterns
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Estimate noise via high-frequency content using Laplacian
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        var_lap = float(lap.var())
        # Normalize variance to [0,1] using a soft scaling; typical var_lap ranges widely
        # Use 1000 as a rough scale and clamp
        score = var_lap / (var_lap + 1000.0)
        return max(0.0, min(1.0, score)) 