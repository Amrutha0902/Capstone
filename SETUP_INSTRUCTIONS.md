# Step-by-Step Setup Instructions

## For Complete Beginners

### Prerequisites Checklist
- [ ] Python 3.10 or higher installed
- [ ] Internet connection (for downloading packages)
- [ ] At least 4GB free disk space
- [ ] Your dataset folder (`OriginalDataset`) ready

---

## Step 1: Install Python

1. **Download Python**:
   - Go to [python.org/downloads](https://www.python.org/downloads/)
   - Download Python 3.10 or newer (3.11 recommended)
   - **IMPORTANT**: During installation, check the box "Add Python to PATH"

2. **Verify Installation**:
   - Open Command Prompt (Windows) or Terminal (Mac/Linux)
   - Type: `python --version`
   - You should see: `Python 3.10.x` or higher

---

## Step 2: Navigate to Project Folder

1. **Open Terminal/Command Prompt**
2. **Navigate to your project folder**:
   ```bash
   cd D:\Capstone
   ```
   (Replace with your actual project path)

---

## Step 3: Create Folder Structure

The following folders should already exist, but verify:

```
CAPSTONE/
├── OriginalDataset/     (Your dataset - should already exist)
├── models/              (Will be created automatically)
├── templates/           (Should exist)
├── static/              (Will be created automatically)
│   ├── uploads/
│   └── output/
```

**If folders don't exist**, create them:
- Windows: Use File Explorer or run: `mkdir models templates static\uploads static\output`
- Mac/Linux: Run: `mkdir -p models templates static/uploads static/output`

---

## Step 4: Install Dependencies

1. **Open terminal in project folder** (same as Step 2)

2. **Install all packages**:
   ```bash
   pip install -r requirements.txt
   ```

   **This will take 10-15 minutes** as it downloads:
   - TensorFlow (large package ~500MB)
   - Other dependencies

3. **If you get errors**:
   - Try: `python -m pip install -r requirements.txt`
   - Or: `pip3 install -r requirements.txt`

---

## Step 5: Train the Model (or Use Pre-trained)

### Option A: Train Using Notebook

1. **Open Jupyter Notebook**:
   ```bash
   pip install jupyter
   jupyter notebook
   ```

2. **Open the notebook**: `image-model-capstone (5).ipynb` (or `image_model_capstone.ipynb` if renamed)

3. **Update the dataset path** in Cell 1:
   ```python
   DATA_PATH = './OriginalDataset'  # Or your full path
   ```

4. **Run all cells** (Cell → Run All)
   - This will take 1-3 hours depending on your computer
   - The model will be saved to `models/best_model.h5` automatically

### Option B: Use Pre-trained Model

If you have a pre-trained model:
1. Place it in the `models/` folder
2. Rename it to `best_model.h5`

---

## Step 6: Run the Flask Application

1. **Make sure you're in the project folder**:
   ```bash
   cd D:\Capstone
   ```

2. **Run the app**:
   ```bash
   python app.py
   ```

3. **You should see**:
   ```
   ============================================================
   Alzheimer's Disease Classification Web App
   ============================================================
   ✅ Predictor initialized
   ✅ LLM Explainer initialized
   
   🚀 Starting Flask server...
   📱 Access the app at: http://127.0.0.1:5000
   ============================================================
   ```

---

## Step 7: Access the Web Interface

1. **Open your web browser** (Chrome, Firefox, Edge, etc.)

2. **Go to**: `http://127.0.0.1:5000`

3. **Upload an image**:
   - Click "Choose File" or drag and drop an MRI image
   - Click "Analyze Image"
   - Wait for results!

---

## Troubleshooting

### "Model not found" Error

**Problem**: `Model file not found at models/best_model.h5`

**Solution**:
1. Train the model using the notebook (Step 5)
2. Or place your trained model in `models/` folder as `best_model.h5`

### "Module not found" Error

**Problem**: `ModuleNotFoundError: No module named 'tensorflow'`

**Solution**:
```bash
pip install -r requirements.txt
```

### Port Already in Use

**Problem**: `Address already in use`

**Solution**: 
1. Close other applications using port 5000
2. Or change port in `app.py` (line 160): `port=5001`

### Python Not Found

**Problem**: `'python' is not recognized`

**Solution**:
1. Reinstall Python and check "Add to PATH"
2. Or use: `py app.py` (Windows)
3. Or use: `python3 app.py` (Mac/Linux)

---

## Optional: Using OpenAI API for Better Explanations

1. **Get API Key**:
   - Sign up at [platform.openai.com](https://platform.openai.com)
   - Create API key

2. **Set Environment Variable**:
   
   **Windows (PowerShell)**:
   ```powershell
   $env:OPENAI_API_KEY="your-api-key-here"
   ```
   
   **Windows (CMD)**:
   ```cmd
   set OPENAI_API_KEY=your-api-key-here
   ```
   
   **Mac/Linux**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. **Or create `.env` file** in project root:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

**Note**: Without API key, the app uses a local model (slower but free).

---

## Testing the System

1. **Health Check**: Visit `http://127.0.0.1:5000/health`
   - Should return JSON with status information

2. **Upload Test Image**:
   - Use any image from your `OriginalDataset` folder
   - Should see:
     - Prediction result
     - Confidence score
     - Grad-CAM heatmap
     - LLM explanation

---

## Next Steps

- **Customize**: Edit `templates/index.html` to change the UI
- **Improve Model**: Retrain with more data or different architectures
- **Deploy**: Follow deployment instructions in README.md

---

## Need Help?

1. Check the main `README.md` for detailed documentation
2. Review error messages in the terminal
3. Verify all files are in correct locations
4. Ensure Python version is 3.10+

---

**You're all set! 🎉**

