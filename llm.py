"""
LLM Module for generating explanations
Uses OpenAI API if available, otherwise falls back to local model
"""

import os
from typing import Dict, Optional

class LLMExplainer:
    """Generates natural language explanations for predictions"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LLM explainer
        
        Args:
            api_key: OpenAI API key (optional, can use environment variable)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.use_openai = self.api_key is not None
        
        if not self.use_openai:
            print("⚠️  OpenAI API key not found. Using local model fallback.")
            try:
                # Try to install tf-keras if needed for compatibility
                try:
                    import tf_keras
                except ImportError:
                    print("⚠️  Installing tf-keras for compatibility...")
                    import subprocess
                    import sys
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "tf-keras", "--quiet"])
                    import tf_keras
                
                from transformers import pipeline
                self.local_model = pipeline(
                    "text2text-generation",
                    model="google/flan-t5-base",
                    device=-1  # CPU
                )
            except Exception as e:
                print(f"⚠️  Could not load local model: {e}")
                print("   Will use fallback explanation instead.")
                self.local_model = None
    
    def generate_explanation(self, prediction_result: Dict, heatmap_info: str = "", 
                            visualization_type: str = "grid") -> str:
        """
        Generate comprehensive explanation for prediction
        
        Args:
            prediction_result: Dictionary with prediction results:
                - predicted_class: str
                - confidence: float
                - all_probabilities: dict
            heatmap_info: Description of heatmap findings
            visualization_type: Type of visualization (grid, saliency, gradcam)
        
        Returns:
            Comprehensive explanation text
        """
        predicted_class = prediction_result.get('predicted_class', 'Unknown')
        confidence = prediction_result.get('confidence', 0.0)
        all_probs = prediction_result.get('all_probabilities', {})
        
        # Format probabilities
        prob_text = "\n".join([
            f"- {cls}: {prob*100:.1f}%" 
            for cls, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        ])
        
        # Create comprehensive prompt
        prompt = f"""You are a medical AI assistant explaining Alzheimer's disease MRI scan classification results. Your explanation should be comprehensive, detailed, and understandable for both healthcare professionals and patients/families.

PREDICTION RESULTS:
- Predicted Class: {predicted_class}
- Confidence: {confidence*100:.1f}%
- All Class Probabilities:
{prob_text}

HEATMAP VISUALIZATION INFORMATION:
{heatmap_info if heatmap_info else "The heatmap shows which brain regions the model focused on for this prediction using Explainable AI (XAI) techniques."}

VISUALIZATION TYPE: {visualization_type}

Please provide a COMPREHENSIVE, DETAILED explanation that covers ALL of the following points:

1. PREDICTION EXPLANATION (For both clinicians and general audience):
   - What does this prediction mean in simple, clear terms?
   - What stage/severity of Alzheimer's disease does this represent?
   - How confident is the model in this prediction?
   - What do the probability scores for all classes tell us?

2. CAUSES AND PATHOPHYSIOLOGY (Explain the underlying causes):
   - What causes Alzheimer's disease at this stage?
   - What biological/neurological changes are occurring in the brain?
   - What happens to brain cells and structures at this stage?
   - Explain in terms understandable to both medical professionals and general public

3. XAI HEATMAP EXPLANATION (Detailed explanation of the visualization):
   - HOW does the heatmap work? (Explain the XAI technique used)
   - WHY are certain brain regions highlighted?
   - WHAT do the different colors/regions represent?
   - Which specific brain areas are most important for this classification?
   - What do these highlighted regions mean clinically?
   - How do these regions relate to Alzheimer's disease pathology?
   - Explain the connection between the heatmap patterns and the prediction

4. CLINICAL SIGNIFICANCE:
   - What does this mean for the patient?
   - What are the typical symptoms at this stage?
   - What brain functions are typically affected?
   - Which brain structures (hippocampus, ventricles, cortex, etc.) are involved?

5. IMPORTANT NOTES:
   - This is an AI-assisted analysis, not a definitive diagnosis
   - The importance of consulting with qualified medical professionals
   - The role of this tool in the diagnostic process

REQUIREMENTS:
- Be thorough and detailed - DO NOT limit yourself to 2-3 sentences
- Explain EVERY point mentioned above
- Use clear language that is accessible to non-medical readers
- Include appropriate medical terminology with explanations for clinicians
- Structure the explanation with clear sections or paragraphs
- Make it comprehensive enough to be educational and informative
- Ensure all technical terms are explained in context

Write a detailed, comprehensive explanation covering all these aspects."""
        
        if self.use_openai:
            return self._generate_openai(prompt, comprehensive=True)
        else:
            return self._generate_local_comprehensive(prediction_result, heatmap_info, visualization_type)
    
    def _generate_openai(self, prompt: str, comprehensive: bool = False) -> str:
        """Generate explanation using OpenAI API"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            # Use higher token limit for comprehensive explanations
            max_tokens = 1500 if comprehensive else 200
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical AI assistant that provides comprehensive, detailed explanations of medical imaging results. You explain complex medical concepts in ways that are accessible to both healthcare professionals and patients/families. You always provide thorough, educational explanations that cover all aspects requested."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️  Error with OpenAI API: {e}")
            return self._generate_comprehensive_fallback(prompt)
    
    def _generate_local_comprehensive(self, prediction_result: Dict, heatmap_info: str, 
                                      visualization_type: str) -> str:
        """Generate comprehensive explanation using local model or detailed fallback"""
        predicted_class = prediction_result.get('predicted_class', 'Unknown')
        confidence = prediction_result.get('confidence', 0.0)
        all_probs = prediction_result.get('all_probabilities', {})
        
        # Create comprehensive fallback explanation since local models may not be able to generate
        # the level of detail needed
        return self._generate_comprehensive_fallback_detailed(
            predicted_class, confidence, all_probs, heatmap_info, visualization_type
        )
    
    def _generate_comprehensive_fallback(self, prompt: str) -> str:
        """Comprehensive fallback explanation if LLM fails"""
        # Try to extract key info from prompt
        predicted_class = "Unknown"
        confidence = "N/A"
        
        if "Predicted Class:" in prompt:
            lines = prompt.split('\n')
            for line in lines:
                if 'Predicted Class:' in line:
                    predicted_class = line.split(':')[1].strip()
                if 'Confidence:' in line:
                    conf_str = line.split(':')[1].strip().replace('%', '')
                    try:
                        confidence = float(conf_str)
                    except:
                        confidence = conf_str
        
        return self._generate_comprehensive_fallback_detailed(
            predicted_class, confidence, {}, "", "grid"
        )
    
    def _generate_comprehensive_fallback_detailed(self, predicted_class: str, confidence, 
                                                  all_probs: Dict, heatmap_info: str,
                                                  visualization_type: str) -> str:
        """Generate detailed comprehensive fallback explanation"""
        
        # Class descriptions
        class_descriptions = {
            'NonDemented': {
                'simple': 'normal brain structure with no signs of dementia',
                'clinical': 'No significant neurodegenerative changes detected. Brain structures appear within normal limits.',
                'causes': 'This classification indicates the absence of Alzheimer\'s disease pathology. The brain shows normal aging patterns without the characteristic changes associated with dementia.',
                'symptoms': 'No cognitive decline or memory issues expected at this stage.',
                'regions': 'All brain regions, including the hippocampus, cortex, and ventricles, appear normal in size and structure.'
            },
            'VeryMildDemented': {
                'simple': 'early, very mild signs of dementia - the earliest detectable stage',
                'clinical': 'Minimal neurodegenerative changes. Early stage Alzheimer\'s disease with subtle structural alterations.',
                'causes': 'Alzheimer\'s disease begins with the accumulation of amyloid-beta plaques and tau protein tangles in the brain. At this very early stage, these changes are minimal but detectable. The disease process involves progressive damage to neurons (brain cells), starting in memory-related regions.',
                'symptoms': 'May experience minor memory lapses, slight difficulty finding words, or subtle changes in thinking that may not significantly impact daily life.',
                'regions': 'Early changes often begin in the hippocampus (memory center) and entorhinal cortex, which are critical for forming new memories.'
            },
            'MildDemented': {
                'simple': 'mild to moderate dementia - noticeable cognitive decline',
                'clinical': 'Moderate neurodegenerative changes. Alzheimer\'s disease is progressing with more pronounced brain atrophy.',
                'causes': 'At this stage, amyloid plaques and tau tangles have spread more extensively. Neurons continue to die, leading to brain shrinkage (atrophy). The brain\'s ability to communicate between regions is disrupted. Neurotransmitters (chemical messengers) are affected, particularly acetylcholine, which is important for memory and learning.',
                'symptoms': 'More noticeable memory loss, difficulty with problem-solving, confusion about time and place, challenges with daily tasks, personality changes, and difficulty with language.',
                'regions': 'Significant involvement of the hippocampus and temporal lobes (memory and language), with spread to parietal lobes (spatial awareness) and frontal lobes (executive function). Ventricular enlargement becomes more apparent.'
            },
            'ModerateDemented': {
                'simple': 'moderate to severe dementia - significant cognitive impairment',
                'clinical': 'Advanced neurodegenerative changes. Severe Alzheimer\'s disease with extensive brain atrophy and structural damage.',
                'causes': 'Widespread neuronal death and brain shrinkage. Amyloid plaques and tau tangles are extensive throughout the brain. The cerebral cortex (the outer layer responsible for thinking, planning, and memory) shows significant thinning. The brain loses its ability to process information efficiently. Blood flow to brain regions may be reduced, and inflammation contributes to further damage.',
                'symptoms': 'Severe memory loss, confusion, difficulty recognizing familiar people, problems with language and communication, difficulty with motor functions, wandering, and significant personality changes. May require assistance with daily activities.',
                'regions': 'Extensive atrophy affecting multiple brain regions: severe hippocampal shrinkage, significant cortical thinning, enlarged ventricles (fluid-filled spaces), and damage to white matter (nerve fibers connecting brain regions).'
            }
        }
        
        desc = class_descriptions.get(predicted_class, class_descriptions['NonDemented'])
        
        # Confidence interpretation
        if isinstance(confidence, (int, float)):
            if confidence >= 0.9:
                conf_text = f"very high ({confidence*100:.1f}%)"
                conf_explanation = "The model is highly confident in this prediction, indicating that the brain scan shows clear, distinctive features characteristic of this stage."
            elif confidence >= 0.7:
                conf_text = f"high ({confidence*100:.1f}%)"
                conf_explanation = "The model shows high confidence, suggesting strong evidence for this classification."
            elif confidence >= 0.5:
                conf_text = f"moderate ({confidence*100:.1f}%)"
                conf_explanation = "The model shows moderate confidence. While there is evidence for this classification, there may be some uncertainty."
            else:
                conf_text = f"lower ({confidence*100:.1f}%)"
                conf_explanation = "The model shows lower confidence, suggesting the brain scan features are less definitive."
        else:
            conf_text = str(confidence)
            conf_explanation = "Confidence level indicates the model's certainty in this classification."
        
        # XAI explanation
        xai_explanation = """
HOW THE HEATMAP WORKS:
The heatmap visualization uses Explainable AI (XAI) techniques, specifically Grad-CAM (Gradient-weighted Class Activation Mapping), to show which parts of the brain image the artificial intelligence model focused on when making its prediction. The model analyzes the image through multiple layers, and the heatmap highlights the regions where changes in the image would most significantly affect the prediction.

WHAT THE COLORS MEAN:
"""
        
        if visualization_type == "grid":
            xai_explanation += """
The 2x2 grid visualization shows four separate heatmaps, one for each possible classification:
- Green regions indicate areas important for "NonDemented" classification (normal brain)
- Yellow regions show areas important for "VeryMildDemented" classification (early changes)
- Orange regions highlight areas important for "MildDemented" classification (moderate changes)
- Red regions indicate areas important for "ModerateDemented" classification (severe changes)

Each panel shows where the model would look if it were predicting that specific stage. The panel with the most intense, widespread coloring typically corresponds to the predicted class.
"""
        else:
            xai_explanation += """
The heatmap uses color intensity to show importance:
- Red and yellow regions: Areas of HIGH importance - the model heavily relies on these regions for the prediction
- Orange regions: Areas of MODERATE importance
- Blue/dark regions: Areas of LOWER importance

Brighter, more intense colors indicate brain regions where the presence of certain features (like tissue shrinkage, ventricular enlargement, or structural changes) most strongly influenced the classification decision.
"""
        
        xai_explanation += f"""
WHY THESE REGIONS MATTER:
The highlighted regions in the heatmap correspond to brain areas known to be affected in Alzheimer's disease:
- Hippocampus: Critical for memory formation; one of the first areas affected
- Temporal lobes: Important for memory, language, and object recognition
- Ventricles: Fluid-filled spaces that enlarge as brain tissue shrinks
- Cortex: The outer brain layer responsible for thinking, planning, and memory
- White matter: Nerve fibers that connect different brain regions

These regions are highlighted because the AI model has learned (from thousands of training images) that changes in these areas are strong indicators of Alzheimer's disease progression. The pattern and intensity of highlights help explain not just WHAT the prediction is, but WHY the model made that prediction.
"""
        
        # Build comprehensive explanation
        explanation = f"""
## PREDICTION EXPLANATION

**What This Means:**
The MRI brain scan has been classified as **{predicted_class}**, which indicates {desc['simple']}. The model's confidence level is {conf_text}. {conf_explanation}

**Understanding the Results:**
From a clinical perspective, {desc['clinical']} This classification is based on analyzing patterns in the brain structure that the artificial intelligence model has learned from extensive training on thousands of brain scans.

**All Class Probabilities:**
"""
        
        # Add probability breakdown
        if all_probs:
            try:
                for cls, prob in sorted(all_probs.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True):
                    # Handle both percentage (0-100) and probability (0-1) formats
                    if isinstance(prob, (int, float)):
                        if prob > 1:  # Already a percentage
                            prob_value = prob
                        else:  # Probability (0-1)
                            prob_value = prob * 100
                        explanation += f"- {cls}: {prob_value:.1f}%\n"
                    else:
                        explanation += f"- {cls}: {prob}\n"
            except Exception as e:
                explanation += "- Probability data available but formatting error occurred\n"
        else:
            explanation += "- Probability data not available\n"
        
        explanation += f"""
The probability distribution shows how likely each classification is according to the model. The predicted class ({predicted_class}) has the highest probability, but the other probabilities provide context about the model's certainty and alternative possibilities.

---

## CAUSES AND UNDERLYING MECHANISMS

**What Causes Alzheimer's Disease at This Stage:**

{desc['causes']}

**Biological Processes:**
Alzheimer's disease is a progressive neurodegenerative disorder characterized by:
1. **Amyloid-Beta Plaques**: Abnormal protein deposits that accumulate between neurons, disrupting cell communication
2. **Tau Protein Tangles**: Twisted fibers inside neurons that interfere with nutrient transport
3. **Neuronal Death**: Brain cells die, leading to brain shrinkage (atrophy)
4. **Inflammation**: The brain's immune response, which can cause additional damage
5. **Reduced Blood Flow**: Decreased oxygen and nutrient supply to brain regions

These processes start years before symptoms appear and progressively worsen, affecting different brain regions at different stages of the disease.

---

## EXPLAINABLE AI (XAI) HEATMAP ANALYSIS

{xai_explanation}

**Clinical Interpretation:**
The heatmap helps clinicians understand:
1. **Which brain regions** are showing the most significant changes
2. **How severe** the changes are in different areas (intensity of color)
3. **The pattern of involvement** (whether changes are localized or widespread)
4. **Progression indicators** (which regions suggest advancing disease)

This information can help in:
- Confirming the diagnosis
- Understanding the stage of disease
- Planning treatment approaches
- Monitoring disease progression over time
- Explaining the condition to patients and families

---

## CLINICAL SIGNIFICANCE AND SYMPTOMS

**What This Means for the Patient:**

At the **{predicted_class}** stage, patients typically experience:
{desc['symptoms']}

**Affected Brain Functions:**
{desc['regions']}

The specific brain regions highlighted in the heatmap correspond to real-world functions:
- **Memory problems** correlate with hippocampal and temporal lobe changes
- **Language difficulties** relate to temporal and frontal lobe involvement  
- **Spatial confusion** connects to parietal lobe changes
- **Executive function decline** (planning, decision-making) links to frontal lobe changes

**Brain Structures Involved:**
- **Hippocampus**: Memory formation and consolidation
- **Cerebral Cortex**: Thinking, planning, language, and perception
- **Ventricles**: Enlargement indicates overall brain shrinkage
- **White Matter**: Connections between brain regions
- **Temporal Lobes**: Memory and language processing
- **Parietal Lobes**: Spatial awareness and attention
- **Frontal Lobes**: Executive functions and personality

---

## IMPORTANT MEDICAL NOTES

**This is an AI-Assisted Analysis:**
This prediction is generated by an artificial intelligence system trained on brain MRI scans. While AI can identify patterns that may indicate Alzheimer's disease, it should be used as a **supplementary tool** in the diagnostic process, not as a replacement for clinical judgment.

**Consultation with Healthcare Professionals:**
- This analysis should be reviewed and interpreted by qualified neurologists, radiologists, or geriatricians
- A definitive diagnosis of Alzheimer's disease requires:
  - Clinical evaluation and medical history
  - Neurological examination
  - Cognitive testing
  - Additional imaging studies if needed
  - Laboratory tests to rule out other conditions
  - Consideration of the patient's overall health and symptoms

**Role in Diagnosis:**
This AI tool can help:
- Provide additional insights to support clinical decision-making
- Identify patterns that may be subtle or early-stage
- Offer a quantitative assessment of brain changes
- Assist in monitoring disease progression over time
- Support research and understanding of Alzheimer's disease

However, the final diagnosis and treatment decisions should always be made by qualified medical professionals who can consider the full clinical picture, including patient history, symptoms, physical examination, and other diagnostic tests.

---

## SUMMARY

The MRI scan analysis indicates **{predicted_class}** with {conf_text} confidence. The heatmap visualization shows which specific brain regions the AI model identified as most important for this classification, helping to explain both the prediction and the underlying brain changes. {desc['causes']}

This comprehensive analysis provides valuable information for understanding the current state of brain health, but it should be interpreted in conjunction with clinical evaluation by healthcare professionals. Early detection and diagnosis can be important for managing symptoms and planning care, but a complete medical assessment is essential for proper diagnosis and treatment.
"""
        
        return explanation.strip()

def get_explainer(api_key: Optional[str] = None) -> LLMExplainer:
    """
    Get or create LLM explainer instance
    
    Args:
        api_key: Optional OpenAI API key
    
    Returns:
        LLMExplainer instance
    """
    return LLMExplainer(api_key)

