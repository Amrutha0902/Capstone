"""
Prediction Module for Alzheimer's Disease Classification
Handles model loading and prediction functionality
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

# Configuration
CLASSES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
TARGET_SIZE = (96, 96)
MODEL_PATH = 'models/best_model.keras'  # Default model path (Keras 3 format)
# Fallback to H5 if .keras doesn't exist
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = 'models/best_model.h5'

class AlzheimerPredictor:
    """Handles model loading and predictions"""
    
    def __init__(self, model_path=None):
        """
        Initialize the predictor with a trained model
        
        Args:
            model_path: Path to the saved model file (.h5)
        """
        self.model = None
        self.model_path = model_path or MODEL_PATH
        self.classes = CLASSES
        self.target_size = TARGET_SIZE
        
    def load_model(self):
        """Load the trained model from disk"""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(
                    f"Model file not found at {self.model_path}. "
                    "Please train the model first using the notebook."
                )
            
            print(f"Loading model from {self.model_path}...")
            
            # Try loading with different methods for Keras 3 compatibility
            try:
                # Method 1: Standard load (works for most cases)
                self.model = keras.models.load_model(self.model_path, compile=False)
            except Exception as e1:
                print(f"Standard load failed: {e1}")
                try:
                    # Method 2: Load with custom InputLayer to handle batch_shape
                    import tensorflow as tf
                    from tensorflow.keras.layers import InputLayer
                    
                    # Custom InputLayer that ignores batch_shape
                    class CompatibleInputLayer(InputLayer):
                        @classmethod
                        def from_config(cls, config):
                            # Remove batch_shape if present (Keras 3 doesn't support it)
                            config = config.copy()
                            if 'batch_shape' in config:
                                # Convert batch_shape to input_shape
                                batch_shape = config.pop('batch_shape')
                                if batch_shape and len(batch_shape) > 1:
                                    config['input_shape'] = batch_shape[1:]
                            return super().from_config(config)
                    
                    self.model = keras.models.load_model(
                        self.model_path,
                        compile=False,
                        custom_objects={'InputLayer': CompatibleInputLayer, 'tf': tf}
                    )
                except Exception as e2:
                    print(f"Custom InputLayer load failed: {e2}")
                    try:
                        # Method 3: Try with safe_mode=False
                        self.model = keras.models.load_model(
                            self.model_path,
                            compile=False,
                            safe_mode=False
                        )
                    except Exception as e3:
                        print(f"Safe mode load failed: {e3}")
                        raise ValueError(f"Could not load model. The model was saved with Keras 2 format. Please retrain the model with current Keras version, or convert the model format. Errors: {str(e1)[:200]}, {str(e2)[:200]}, {str(e3)[:200]}")
            
            # Recompile the model if needed
            try:
                if not hasattr(self.model, 'optimizer') or self.model.optimizer is None:
                    self.model.compile(
                        optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                    )
            except Exception as e:
                print(f"Warning: Could not recompile model: {e}")
                print("Model will work for inference but may have issues with training.")
            
            # Test the model with a dummy input to ensure it works
            try:
                dummy_input = np.zeros((1, 96, 96, 3), dtype=np.float32)
                _ = self.model.predict(dummy_input, verbose=0)
                print("[OK] Model loaded and tested successfully!")
            except Exception as e:
                print(f"[WARNING] Model loaded but test prediction failed: {e}")
                print("   Model may still work, but there might be compatibility issues.")
            
            return True
        except Exception as e:
            print(f"[ERROR] Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def preprocess_image(self, image_path):
        """
        Preprocess an image for prediction
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array ready for model input
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image from {image_path}")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to target size
            img = cv2.resize(img, self.target_size)
            
            # Normalize to [0, 1]
            img = img.astype('float32') / 255.0
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            return img
        except Exception as e:
            raise ValueError(f"Error preprocessing image: {str(e)}")
    
    def predict(self, image_path):
        """
        Make a prediction on an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            dict with prediction results:
            {
                'predicted_class': str,
                'confidence': float,
                'all_probabilities': dict,
                'class_index': int
            }
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess image
        img_array = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_index])
        predicted_class = self.classes[predicted_index]
        
        # Get all probabilities
        all_probabilities = {
            self.classes[i]: float(predictions[0][i])
            for i in range(len(self.classes))
        }
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_probabilities': all_probabilities,
            'class_index': int(predicted_index)
        }
    
    def get_model_summary(self):
        """Get model architecture summary"""
        if self.model is None:
            return "Model not loaded"
        return str(self.model.summary())

# Global predictor instance (singleton pattern)
_predictor_instance = None

def get_predictor(model_path=None):
    """
    Get or create the global predictor instance
    
    Args:
        model_path: Optional path to model file
        
    Returns:
        AlzheimerPredictor instance
    """
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = AlzheimerPredictor(model_path)
        _predictor_instance.load_model()
    return _predictor_instance

