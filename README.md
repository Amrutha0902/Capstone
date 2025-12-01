# Alzheimer's Disease Stage Classification System 🧠

A deep learning-based diagnostic system for classifying Alzheimer's disease stages from MRI brain scans. The system leverages custom **Convolutional Neural Networks (CNN)** and **VGG architectures** to achieve high accuracy in multi-class classification.

---

## 💡 Overview

This project implements an end-to-end solution for Alzheimer's disease stage prediction using brain MRI images. The system classifies scans into four distinct stages:

* **NonDemented** - No signs of dementia
* **VeryMildDemented** - Early stage with subtle cognitive changes
* **MildDemented** - Moderate cognitive impairment
* **ModerateDemented** - Significant cognitive decline

## ✨ Key Features

### Deep Learning Classification
* Custom **CNN architecture** optimized for brain MRI analysis
* **VGG-based transfer learning** for enhanced feature extraction
* Multi-class classification with confidence scores
* Image preprocessing pipeline with normalization and resizing

### Explainable AI (XAI) Integration
* **Grad-CAM Visualization:** Gradient-weighted Class Activation Mapping to highlight influential brain regions 
* **Saliency Maps:** Pixel-level importance visualization showing which areas drive predictions
* **Multi-class Grid Visualization:** Simultaneous display of class-specific activation patterns
* Transparent decision-making process for clinical interpretability

### LLM-Powered Explanations
* Natural language explanations of prediction results
* Support for **OpenAI API** integration
* Local model fallback using **Flan-T5**
* Comprehensive reports suitable for both clinicians and patients

### Web Application Interface
* **Flask-based** responsive web interface
* Drag-and-drop image upload functionality
* Real-time prediction with visualization
* Interactive result display with probability distributions

---

## 📂 Project Structure

AlzheimersPrediction/ ├── app.py # Flask web application ├── predict.py # Prediction module with CNN inference ├── xai.py # Explainable AI visualization module ├── llm.py # LLM integration for explanations ├── models/ │   └── best_model.keras # Trained classification model ├── templates/ │   └── index.html # Web interface template ├── static/ │   ├── uploads/ # User uploaded images │   └── output/ # Generated visualizations ├── OriginalDataset/ # Training dataset │   ├── NonDemented/ │   ├── VeryMildDemented/ │   ├── MildDemented/ │   └── ModerateDemented/ └── requirements.txt # Python dependencies


---

## ⚙️ Technical Architecture

### Model Architecture
The classification model uses a deep convolutional neural network with the following components:

* Input layer accepting $96 \times 96 \times 3$ RGB images
* Multiple convolutional blocks with batch normalization
* MaxPooling layers for spatial reduction
* Dense layers with dropout regularization
* Softmax output layer for 4-class classification

### XAI Pipeline
The explainability module generates multiple visualization types:

* Standard Grad-CAM: Single-class activation highlighting
* Grid Visualization: $2 \times 2$ grid showing activations for all classes
* Saliency Maps: Gradient-based pixel importance

### Prediction Pipeline
1.  Image preprocessing (resize, normalize)
2.  Model inference with probability scores
3.  XAI visualization generation
4.  LLM explanation synthesis
5.  Response assembly with all components

---

## 🚀 Installation

### Prerequisites
* **Python 3.10** or higher
* `pip` package manager
* 4GB minimum disk space

### Setup Steps

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Amrutha0902/AlzheimersPrediction.git](https://github.com/Amrutha0902/AlzheimersPrediction.git)
    cd AlzheimersPrediction
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the application:**
    ```bash
    python app.py
    ```
4.  **Access the web interface** at `http://127.0.0.1:5000`

## 💻 Usage

### Web Interface
1.  Navigate to the application URL.
2.  Upload an MRI brain scan image.
3.  Click **"Analyze Image"**.
4.  View the prediction results including:
    * Classification label and confidence
    * Probability distribution across all classes
    * Grad-CAM heatmap visualization
    * Natural language explanation

### API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Main web interface |
| `/predict` | `POST` | Image classification endpoint |
| `/health` | `GET` | System health check |

## 🛠️ Configuration

### Environment Variables

| Variable | Description |
| :--- | :--- |
| `OPENAI_API_KEY` | Optional API key for enhanced LLM explanations |

### Model Settings
* `TARGET_SIZE`: Image input dimensions (default: $96 \times 96$)
* `MODEL_PATH`: Path to trained model file

---

## 📈 Development Status

This project is under active development with the following roadmap:

### Completed Tasks ✅

* [x] CNN-based image classification
* [x] VGG architecture integration
* [x] Grad-CAM XAI visualizations
* [x] Flask web application
* [x] LLM explanation generation

### In Progress ⏳

* [ ] Enhanced XAI feature integration
* [ ] Advanced LLM prompt optimization
* [ ] Model performance improvements

### Planned 💡

* [ ] Speech-based diagnostic model integration
* [ ] Multi-modal analysis combining imaging and audio
* [ ] Clinical validation studies
* [ ] Mobile application development

### Dependencies
Key libraries used in this project:

* **TensorFlow/Keras** for deep learning
* **OpenCV** for image processing
* **Flask** for web framework
* **Transformers** for local LLM support
* **Matplotlib** for visualization

See `requirements.txt` for the complete dependency list.

## 🙏 Acknowledgments

This project was developed as a capstone project focusing on applying deep learning techniques to medical image analysis for Alzheimer's disease diagnosis support.
