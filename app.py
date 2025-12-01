"""
Flask Web Application for Alzheimer's Disease Image Classification
Main application file
"""

import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import uuid
from datetime import datetime

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, continue without it
    pass

from predict import get_predictor
from xai import (GradCAM, create_gradcam, create_multi_class_gradcam,
                 create_saliency_visualization, create_grid_visualization)
from llm import get_explainer

# Configuration
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/output'

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Initialize components
predictor = None
explainer = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def init_components():
    """Initialize predictor and explainer"""
    global predictor, explainer
    
    try:
        # Initialize predictor
        predictor = get_predictor()
        print("[OK] Predictor initialized")
        
        # Initialize explainer
        explainer = get_explainer()
        print("[OK] LLM Explainer initialized")
        
        return True
    except Exception as e:
        print(f"[ERROR] Error initializing components: {e}")
        return False

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload an image (PNG, JPG, JPEG)'}), 400
        
        # Check if components are initialized
        if predictor is None:
            return jsonify({'error': 'Predictor not initialized. Please restart the app.'}), 500
        
        if predictor.model is None:
            return jsonify({'error': 'Model not loaded. Please check model file exists at models/best_model.h5'}), 500
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Make prediction
        try:
            prediction_result = predictor.predict(filepath)
        except Exception as e:
            print(f"[ERROR] Error making prediction: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
        
        # Generate multiple XAI visualizations with different techniques
        heatmap_filename = f"heatmap_grid_{uuid.uuid4()}.png"  # Grid view (most visible)
        heatmap_path = os.path.join(app.config['OUTPUT_FOLDER'], heatmap_filename)
        
        saliency_filename = f"heatmap_saliency_{uuid.uuid4()}.png"  # Saliency map
        saliency_path = os.path.join(app.config['OUTPUT_FOLDER'], saliency_filename)
        
        # Also generate single-class heatmap for backward compatibility
        single_heatmap_filename = f"heatmap_single_{uuid.uuid4()}.png"
        single_heatmap_path = os.path.join(app.config['OUTPUT_FOLDER'], single_heatmap_filename)
        
        try:
            # Ensure model is built before creating Grad-CAM
            # The model should already be built from the prediction above
            if predictor.model is not None:
                # Get predictions for multi-class visualization
                img_array = predictor.preprocess_image(filepath)
                predictions = predictor.model.predict(img_array, verbose=0)
                
                print(f"[INFO] Starting XAI visualization generation...")
                print(f"[INFO] Predictions: {[f'{p*100:.1f}%' for p in predictions[0]]}")
                
                # Try grid visualization first (most visible, shows all 4 classes separately)
                try:
                    create_grid_visualization(
                        predictor.model,
                        filepath,
                        heatmap_path,
                        predictions=predictions
                    )
                    
                    if os.path.exists(heatmap_path):
                        file_size = os.path.getsize(heatmap_path)
                        print(f"[OK] Grid visualization generated: {heatmap_filename} ({file_size} bytes)")
                    else:
                        raise FileNotFoundError(f"Grid visualization file not found")
                        
                except Exception as e_grid:
                    print(f"[WARNING] Grid visualization failed: {e_grid}")
                    # Try saliency map as fallback
                    try:
                        create_saliency_visualization(
                            predictor.model,
                            filepath,
                            heatmap_path,
                            class_index=prediction_result['class_index'],
                            predictions=predictions
                        )
                        if os.path.exists(heatmap_path):
                            file_size = os.path.getsize(heatmap_path)
                            print(f"[OK] Saliency map generated: {heatmap_filename} ({file_size} bytes)")
                        else:
                            heatmap_filename = None
                    except Exception as e_sal:
                        print(f"[WARNING] Saliency map also failed: {e_sal}")
                        heatmap_filename = None
                
                # Also generate saliency map separately
                try:
                    create_saliency_visualization(
                        predictor.model,
                        filepath,
                        saliency_path,
                        class_index=prediction_result['class_index'],
                        predictions=predictions
                    )
                    saliency_filename = saliency_filename if os.path.exists(saliency_path) else None
                except:
                    saliency_filename = None
                
                # Also create single-class heatmap for the predicted class
                if heatmap_filename:  # Only if multi-class succeeded
                    try:
                        create_gradcam(
                            predictor.model,
                            filepath,
                            single_heatmap_path,
                            class_index=prediction_result['class_index']
                        )
                        if os.path.exists(single_heatmap_path):
                            print(f"[OK] Single-class heatmap generated: {single_heatmap_filename}")
                        else:
                            single_heatmap_filename = None
                    except Exception as e2:
                        print(f"[WARNING] Could not generate single-class heatmap: {e2}")
                        single_heatmap_filename = None
            else:
                raise ValueError("Model is None, cannot generate heatmap")
        except Exception as e:
            print(f"[WARNING] Error generating heatmap: {e}")
            import traceback
            traceback.print_exc()
            heatmap_filename = None
            single_heatmap_filename = None
            # Continue without heatmap - prediction still works
        
        # Generate comprehensive LLM explanation
        # Determine visualization type and create detailed heatmap info
        visualization_type = "grid" if heatmap_filename and "grid" in heatmap_filename else "saliency" if saliency_filename else "gradcam"
        
        # Create detailed heatmap information
        if heatmap_filename and "grid" in heatmap_filename:
            heatmap_info = f"""A 2x2 grid visualization has been generated showing class-specific activation maps for all four Alzheimer's disease stages using Grad-CAM (Gradient-weighted Class Activation Mapping):
- Top-left (Green): NonDemented regions - shows areas indicating normal brain structure
- Top-right (Yellow): VeryMildDemented regions - highlights areas of early pathological changes
- Bottom-left (Orange): MildDemented regions - indicates regions with moderate brain changes
- Bottom-right (Red): ModerateDemented regions - shows areas with severe changes

Each panel uses Grad-CAM technique to highlight which brain regions are important for that specific class. The colored overlays show where the model detected features characteristic of each disease stage. Brighter and more intense colors indicate stronger activation and higher relevance to that classification. The model analyzes gradients in convolutional layers to determine which image regions most influence the prediction for each class."""
        elif saliency_filename:
            heatmap_info = """A saliency map visualization has been generated showing pixel-level importance. This gradient-based technique computes how much each pixel contributes to the prediction. Red and yellow regions indicate pixels that most strongly influence the classification decision, while blue areas show less important regions. The saliency map provides a direct visualization of pixel-level sensitivity, showing exactly which pixels the model considers most critical."""
        else:
            heatmap_info = """Grad-CAM heatmap visualization highlights brain regions that influenced the prediction by analyzing gradients in the convolutional neural network layers. The colored regions (red/yellow for high importance) show where the model focused attention, typically corresponding to brain structures like the hippocampus, ventricles, and cortical regions relevant to Alzheimer's disease assessment."""
        
        try:
            explanation = explainer.generate_explanation(
                prediction_result, 
                heatmap_info=heatmap_info,
                visualization_type=visualization_type
            )
        except Exception as e:
            print(f"[WARNING] Error generating explanation: {e}")
            import traceback
            traceback.print_exc()
            # Use comprehensive fallback - access through public method
            try:
                # Try to get the detailed explanation using the explainer's internal method
                # If that fails, use a simpler fallback
                explanation = explainer._generate_comprehensive_fallback_detailed(
                    prediction_result['predicted_class'],
                    prediction_result['confidence'],
                    prediction_result.get('all_probabilities', {}),
                    heatmap_info,
                    visualization_type
                )
            except Exception as fallback_error:
                print(f"[ERROR] Comprehensive fallback also failed: {fallback_error}")
                # Ultimate fallback
                explanation = f"""The MRI scan has been classified as {prediction_result['predicted_class']} with {prediction_result['confidence']*100:.1f}% confidence.

Please consult with a qualified healthcare professional for a complete diagnosis and explanation of the results."""
        
        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'class': prediction_result['predicted_class'],
                'confidence': round(prediction_result['confidence'] * 100, 2),
                'all_probabilities': {
                    k: round(v * 100, 2) 
                    for k, v in prediction_result['all_probabilities'].items()
                }
            },
            'heatmap': f'/static/output/{heatmap_filename}' if heatmap_filename else None,
            'heatmap_saliency': f'/static/output/{saliency_filename}' if saliency_filename else None,
            'heatmap_single': f'/static/output/{single_heatmap_filename}' if single_heatmap_filename else None,
            'original_image': f'/static/uploads/{unique_filename}',
            'explanation': explanation,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"[ERROR] Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'model_loaded': predictor is not None and predictor.model is not None,
        'explainer_ready': explainer is not None
    }
    return jsonify(status)

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/static/output/<filename>')
def output_file(filename):
    """Serve output files"""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    print("=" * 60)
    print("Alzheimer's Disease Classification Web App")
    print("=" * 60)
    
    # Initialize components
    if not init_components():
        print("[WARNING] Some components failed to initialize")
        print("   The app will still start, but predictions may not work.")
    
    print("\n[STARTING] Flask server...")
    print("[INFO] Access the app at: http://127.0.0.1:5000")
    print("=" * 60)
    
    # Run app
    # Note: If port 5000 is busy, change the port number below
    # Or run stop_port.bat to free port 5000
    # Debug mode causes Flask to restart, which may cause model loading issues
    # Set to False if you experience model loading problems on restart
    app.run(debug=False, host='0.0.0.0', port=5000)

