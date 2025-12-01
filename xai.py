"""
Explainable AI (XAI) Module using Grad-CAM, Saliency Maps, and Class-Specific Visualizations
Generates heatmaps showing which parts of the image influenced the prediction
Supports multiple XAI techniques:
- Grad-CAM: Gradient-weighted Class Activation Mapping
- Saliency Maps: Simple gradient-based visualization
- Multi-class visualizations with distinct colors
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
import os

class GradCAM:
    """Grad-CAM implementation for model interpretability"""
    
    def __init__(self, model, last_conv_layer_name=None):
        """
        Initialize Grad-CAM
        
        Args:
            model: Trained Keras model
            last_conv_layer_name: Name of the last convolutional layer.
                                 If None, will be auto-detected.
        """
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name or self._find_last_conv_layer()
        
        if self.last_conv_layer_name is None:
            raise ValueError("No Conv2D layer found in model!")
        
        print(f"Using last conv layer: {self.last_conv_layer_name}")
        
        # Build the model by running a dummy prediction to ensure layers are built
        # This fixes the "layer has never been called" error
        try:
            # Get input shape from model
            if hasattr(model, 'input_shape') and model.input_shape:
                input_shape = model.input_shape[1:]  # Skip batch dimension
            else:
                # Default input shape
                input_shape = (96, 96, 3)
            
            dummy_input = tf.zeros((1,) + input_shape, dtype=tf.float32)
            # Call the model to build it
            _ = model(dummy_input, training=False)
        except Exception as e:
            print(f"Warning: Could not build model with dummy input: {e}")
            # Try alternative: just call model.build if available
            try:
                if hasattr(model, 'build') and not model.built:
                    model.build(input_shape=(None, 96, 96, 3))
                    # Call it again after building
                    dummy_input = tf.zeros((1, 96, 96, 3), dtype=tf.float32)
                    _ = model(dummy_input, training=False)
            except Exception as e2:
                print(f"Warning: Could not build model: {e2}")
        
        # Create gradient model - access layer after model is built
        # The model should now be built from the dummy input above
        try:
            conv_layer = model.get_layer(self.last_conv_layer_name)
            
            # Get model inputs - handle both functional and sequential models
            if hasattr(model, 'input') and model.input is not None:
                model_inputs = model.input
            elif hasattr(model, 'inputs') and model.inputs and len(model.inputs) > 0:
                model_inputs = model.inputs[0] if isinstance(model.inputs, list) else model.inputs
            else:
                # Create input based on input shape
                model_inputs = keras.Input(shape=input_shape, name='input')
            
            # Get model outputs - now that model has been called, output should be available
            # If not, we'll create a wrapper model
            try:
                if hasattr(model, 'output') and model.output is not None:
                    model_output = model.output
                elif hasattr(model, 'outputs') and model.outputs and len(model.outputs) > 0:
                    model_output = model.outputs[0] if isinstance(model.outputs, list) else model.outputs
                else:
                    # Create a wrapper model that calls the original model
                    # This ensures the model is properly called
                    wrapper_output = model(model_inputs)
                    model_output = wrapper_output
            except Exception as e:
                # If accessing output fails, create wrapper
                print(f"Note: Creating wrapper model for output access: {e}")
                wrapper_output = model(model_inputs)
                model_output = wrapper_output
            
            # Get conv layer output - ensure it's accessible
            try:
                conv_output = conv_layer.output
            except Exception:
                # If conv layer output not accessible, we need to rebuild
                # This shouldn't happen if model was called properly
                raise ValueError(f"Could not access output from layer {self.last_conv_layer_name}. Model may not be properly built.")
            
            # Create gradient model
            self.grad_model = keras.models.Model(
                inputs=model_inputs,
                outputs=[conv_output, model_output]
            )
        except Exception as e:
            # Final fallback: create wrapper model
            print(f"Warning creating grad_model: {e}")
            print("Attempting to create wrapper model...")
            try:
                # Create new input with proper shape
                model_inputs = keras.Input(shape=input_shape, name='input')
                
                # Call the original model to get outputs
                # This creates a functional wrapper around the Sequential model
                model_output = model(model_inputs)
                
                # Now we need to get the conv layer output
                # Since we can't easily get intermediate outputs from Sequential,
                # we'll create a model that outputs both the conv layer and final output
                # by creating a custom model
                
                # Get the conv layer
                conv_layer = model.get_layer(self.last_conv_layer_name)
                
                # Create a model that extracts conv layer output
                # We need to trace through the model to get to the conv layer
                # For Sequential models, we can create a model up to that layer
                layer_index = None
                for i, layer in enumerate(model.layers):
                    if layer.name == self.last_conv_layer_name:
                        layer_index = i
                        break
                
                if layer_index is None:
                    raise ValueError(f"Could not find layer {self.last_conv_layer_name}")
                
                # Create a model that outputs up to the conv layer
                intermediate_model = keras.models.Model(
                    inputs=model.input,
                    outputs=conv_layer.output
                )
                
                # Get conv output by calling intermediate model
                conv_output = intermediate_model(model_inputs)
                
                # Create gradient model
                self.grad_model = keras.models.Model(
                    inputs=model_inputs,
                    outputs=[conv_output, model_output]
                )
                print("Successfully created gradient model using wrapper method")
            except Exception as e2:
                # Last resort: try to create intermediate model using original model's structure
                print(f"Error in wrapper method: {e2}")
                try:
                    # If model.input is available, use it
                    if hasattr(model, 'input') and model.input is not None:
                        original_input = model.input
                    else:
                        original_input = keras.Input(shape=input_shape)
                    
                    # Create intermediate model that outputs conv layer
                    intermediate_output = model.get_layer(self.last_conv_layer_name).output
                    intermediate_model = keras.models.Model(
                        inputs=original_input,
                        outputs=intermediate_output
                    )
                    
                    # Now create the gradient model with new input
                    new_input = keras.Input(shape=input_shape, name='gradcam_input')
                    conv_output = intermediate_model(new_input)
                    final_output = model(new_input)
                    
                    self.grad_model = keras.models.Model(
                        inputs=new_input,
                        outputs=[conv_output, final_output]
                    )
                    print("Successfully created gradient model using intermediate model method")
                except Exception as e3:
                    print(f"Error in final fallback: {e3}")
                    import traceback
                    traceback.print_exc()
                    raise ValueError(f"Could not create gradient model. The model may not be compatible with Grad-CAM. Please ensure the model is properly loaded, has been called at least once, and contains Conv2D layers. Error: {e3}")
    
    def _find_last_conv_layer(self):
        """Automatically find the last convolutional layer"""
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        return None
    
    def make_heatmap(self, img_array, class_index=None):
        """
        Generate Grad-CAM heatmap for an image
        
        Args:
            img_array: Preprocessed image array (with batch dimension)
            class_index: Index of class to generate heatmap for.
                        If None, uses predicted class.
        
        Returns:
            Heatmap as numpy array
        """
        # Ensure input is a tensor and watch it
        if not isinstance(img_array, tf.Tensor):
            img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
        
        # Ensure model is built
        if not hasattr(self.grad_model, 'built') or not self.grad_model.built:
            _ = self.grad_model(img_array, training=False)
        
        with tf.GradientTape(watch_accessed_variables=True) as tape:
            # Watch the input tensor explicitly
            tape.watch(img_array)
            
            # Get outputs
            conv_outputs, predictions = self.grad_model(img_array, training=False)
            
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            
            # Get the score for the predicted class
            loss = predictions[:, class_index]
        
        # Compute gradients with respect to conv_outputs
        grads = tape.gradient(loss, conv_outputs)
        
        # Check if gradients are None
        if grads is None:
            raise ValueError("Gradients are None. Model may not be trainable or layer not found.")
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the feature maps by gradients
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        # Normalize heatmap
        heatmap_np = heatmap.numpy()
        heatmap_np = np.maximum(heatmap_np, 0)
        if np.max(heatmap_np) > 0:
            heatmap_np = heatmap_np / np.max(heatmap_np)
        
        return heatmap_np
    
    def generate_visualization(self, img_array, original_image_path=None, 
                              class_index=None, save_path=None):
        """
        Generate complete visualization with heatmap overlay
        
        Args:
            img_array: Preprocessed image array (with batch dimension)
            original_image_path: Path to original image (for display)
            class_index: Class index for heatmap generation
            save_path: Path to save the visualization
        
        Returns:
            Superimposed image array
        """
        # Generate heatmap
        heatmap = self.make_heatmap(img_array, class_index)
        
        # Load original image if path provided
        if original_image_path:
            img = cv2.imread(original_image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (96, 96))
        else:
            # Use preprocessed image (denormalize)
            img = (img_array[0] * 255).astype('uint8')
        
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Superimpose heatmap on image
        superimposed = cv2.addWeighted(
            img.astype('uint8'), 0.6,
            heatmap_colored, 0.4, 0
        )
        
        # Save if path provided
        if save_path:
            plt.imsave(save_path, superimposed)
            print(f"[OK] Heatmap saved to {save_path}")
        
        return superimposed
    
    def save_heatmap_only(self, img_array, save_path, class_index=None):
        """
        Save only the heatmap (without overlay)
        
        Args:
            img_array: Preprocessed image array
            save_path: Path to save heatmap
            class_index: Class index for heatmap
        """
        heatmap = self.make_heatmap(img_array, class_index)
        heatmap_resized = cv2.resize(heatmap, (96, 96))
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), 
            cv2.COLORMAP_JET
        )
        plt.imsave(save_path, heatmap_colored)
        print(f"✅ Heatmap saved to {save_path}")


class MultiClassGradCAM:
    """
    Multi-Class Grad-CAM visualization
    Generates separate heatmaps for each class with distinct colors
    """
    
    # Color mapping for each class
    CLASS_COLORS = {
        0: (0, 255, 0),      # Green - NonDemented
        1: (255, 255, 0),    # Yellow - VeryMildDemented
        2: (255, 165, 0),   # Orange - MildDemented
        3: (255, 0, 0),     # Red - ModerateDemented
    }
    
    CLASS_NAMES = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
    
    def __init__(self, model, last_conv_layer_name=None):
        """
        Initialize Multi-Class Grad-CAM
        
        Args:
            model: Trained Keras model
            last_conv_layer_name: Name of the last convolutional layer
        """
        self.model = model
        self.gradcam = GradCAM(model, last_conv_layer_name)
        self.last_conv_layer_name = self.gradcam.last_conv_layer_name
    
    def generate_class_heatmaps(self, img_array, predictions=None):
        """
        Generate heatmaps for all classes
        
        Args:
            img_array: Preprocessed image array (with batch dimension)
            predictions: Optional predictions array (if None, will compute)
        
        Returns:
            dict with class_index as key and heatmap array as value
        """
        if predictions is None:
            predictions = self.model.predict(img_array, verbose=0)
        
        heatmaps = {}
        for class_idx in range(4):
            try:
                print(f"Generating heatmap for class {class_idx} ({self.CLASS_NAMES[class_idx]})...")
                heatmap = self.gradcam.make_heatmap(img_array, class_index=class_idx)
                if heatmap is not None and heatmap.size > 0:
                    heatmaps[class_idx] = heatmap
                    print(f"  ✓ Heatmap generated for {self.CLASS_NAMES[class_idx]}")
                else:
                    print(f"  ⚠ Empty heatmap for {self.CLASS_NAMES[class_idx]}, using zeros")
                    heatmaps[class_idx] = np.zeros((96, 96))
            except Exception as e:
                print(f"  ✗ Warning: Could not generate heatmap for class {class_idx} ({self.CLASS_NAMES[class_idx]}): {e}")
                import traceback
                traceback.print_exc()
                # Create empty heatmap as fallback
                heatmaps[class_idx] = np.zeros((96, 96))
        
        return heatmaps
    
    def create_multi_class_visualization(self, img_array, original_image_path=None, 
                                       predictions=None, save_path=None, 
                                       threshold=0.3):
        """
        Create a multi-class visualization with different colors for each class
        
        Args:
            img_array: Preprocessed image array
            original_image_path: Path to original image
            predictions: Optional predictions array
            save_path: Path to save visualization
            threshold: Minimum activation threshold to show (0-1)
        
        Returns:
            Combined visualization array
        """
        # Generate heatmaps for all classes
        heatmaps = self.generate_class_heatmaps(img_array, predictions)
        
        # Load original image
        if original_image_path:
            img = cv2.imread(original_image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (96, 96))
        else:
            img = (img_array[0] * 255).astype('uint8')
        
        # Create separate colored heatmaps for each class
        # Normalize each independently so all are visible regardless of prediction confidence
        colored_heatmaps = []
        max_activations = []
        
        for class_idx in range(4):
            heatmap = heatmaps[class_idx]
            heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            
            # Normalize each heatmap independently to [0, 1]
            # This ensures all classes are visible even if prediction is 100% for one
            if np.max(heatmap_resized) > 0:
                heatmap_normalized = heatmap_resized / np.max(heatmap_resized)
            else:
                heatmap_normalized = heatmap_resized
            
            # Apply threshold to focus on important regions
            heatmap_thresholded = np.maximum(heatmap_normalized - threshold, 0)
            if np.max(heatmap_thresholded) > 0:
                heatmap_thresholded = heatmap_thresholded / np.max(heatmap_thresholded)
            
            max_activations.append(np.max(heatmap_thresholded))
            
            # Get color for this class (RGB)
            color = np.array(self.CLASS_COLORS[class_idx], dtype=np.float32) / 255.0
            
            # Create colored heatmap for this class
            colored_hm = np.zeros_like(img, dtype=np.float32)
            for c in range(3):
                colored_hm[:, :, c] = heatmap_thresholded * color[c]
            
            colored_heatmaps.append(colored_hm)
        
        # Combine all colored heatmaps
        # Use equal weighting for all classes to ensure all colors are visible
        # This shows where each class activates, regardless of prediction confidence
        overlay = np.zeros_like(img, dtype=np.float32)
        
        # Give each class equal visual weight to show all simultaneously
        # This is the key: normalize each independently, then combine equally
        for class_idx in range(4):
            # Each class contributes equally to the final visualization
            # This ensures all 4 colors are visible, not just the predicted one
            overlay += colored_heatmaps[class_idx] * 0.25  # Equal 25% weight each
        
        # Normalize to [0, 1] range
        overlay_max = np.max(overlay)
        if overlay_max > 0:
            overlay = overlay / overlay_max
        
        # Enhance colors
        overlay = np.power(overlay, 0.75)  # Gamma correction
        
        # Blend with original image
        img_float = img.astype(np.float32) / 255.0
        alpha = 0.7  # Strong overlay for visibility
        
        # Use overlay blending mode for better color mixing
        mask = np.sum(overlay, axis=2) > 0.01  # Where there's any activation
        result = img_float.copy()
        result[mask] = img_float[mask] * (1 - alpha) + overlay[mask] * alpha
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        # Create legend
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(result)
        ax.axis('off')
        
        # Add legend
        legend_elements = []
        for class_idx in range(4):
            color = np.array(self.CLASS_COLORS[class_idx]) / 255.0
            legend_elements.append(
                Rectangle((0, 0), 1, 1, facecolor=color, 
                         label=self.CLASS_NAMES[class_idx], edgecolor='white', linewidth=2)
            )
        
        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(0.98, 0.98), fontsize=10, framealpha=0.9)
        
        # Add predictions if available
        if predictions is not None:
            pred_text = "Predictions:\n"
            for class_idx in range(4):
                prob = predictions[0][class_idx] * 100
                pred_text += f"{self.CLASS_NAMES[class_idx]}: {prob:.1f}%\n"
            
            ax.text(0.02, 0.98, pred_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                   color='white', family='monospace')
        
        if save_path:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, bbox_inches='tight', dpi=150, pad_inches=0.1, facecolor='white')
                plt.close('all')  # Close all figures to free memory
                
                # Verify file was created
                if os.path.exists(save_path):
                    file_size = os.path.getsize(save_path)
                    print(f"[OK] Multi-class visualization saved to {save_path} ({file_size} bytes)")
                else:
                    raise FileNotFoundError(f"File was not created at {save_path}")
            except Exception as e:
                print(f"[ERROR] Failed to save visualization: {e}")
                plt.close('all')
                raise
        else:
            plt.close('all')
        
        return result
    
    def create_separate_class_visualizations(self, img_array, original_image_path=None,
                                            predictions=None, output_dir=None):
        """
        Create separate visualizations for each class
        
        Args:
            img_array: Preprocessed image array
            original_image_path: Path to original image
            predictions: Optional predictions array
            output_dir: Directory to save individual visualizations
        
        Returns:
            dict with class_index as key and visualization path as value
        """
        heatmaps = self.generate_class_heatmaps(img_array, predictions)
        
        # Load original image
        if original_image_path:
            img = cv2.imread(original_image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (96, 96))
        else:
            img = (img_array[0] * 255).astype('uint8')
        
        saved_paths = {}
        
        for class_idx in range(4):
            heatmap = heatmaps[class_idx]
            heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap_resized = np.uint8(255 * heatmap_resized)
            
            # Apply class-specific color
            color = self.CLASS_COLORS[class_idx]
            
            # Create colored heatmap
            colored_heatmap = np.zeros_like(img)
            for c in range(3):
                colored_heatmap[:, :, c] = heatmap_resized * color[c] / 255.0
            
            # Blend with original
            alpha = 0.5
            result = cv2.addWeighted(
                img.astype(np.uint8), 1 - alpha,
                colored_heatmap.astype(np.uint8), alpha, 0
            )
            
            if output_dir:
                class_name = self.CLASS_NAMES[class_idx]
                save_path = os.path.join(output_dir, f"heatmap_{class_name.lower()}.png")
                plt.imsave(save_path, result)
                saved_paths[class_idx] = save_path
        
        return saved_paths


class SaliencyMap:
    """
    Saliency Map visualization - simpler and often more visible than Grad-CAM
    Shows pixel-level importance using gradients
    """
    
    def __init__(self, model):
        self.model = model
    
    def generate_saliency(self, img_array, class_index=None):
        """
        Generate saliency map for an image
        
        Args:
            img_array: Preprocessed image array (with batch dimension)
            class_index: Class index for saliency (if None, uses predicted)
        
        Returns:
            Saliency map as numpy array
        """
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            predictions = self.model(img_tensor, training=False)
            
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            
            score = predictions[:, class_index]
        
        # Compute gradients
        grads = tape.gradient(score, img_tensor)
        
        # Take absolute value and max across channels
        grads = tf.abs(grads)
        saliency = tf.reduce_max(grads, axis=-1)[0]
        
        # Normalize
        saliency_np = saliency.numpy()
        if np.max(saliency_np) > 0:
            saliency_np = saliency_np / np.max(saliency_np)
        
        return saliency_np


def create_saliency_visualization(model, image_path, output_path, class_index=None, predictions=None):
    """
    Create saliency map visualization with high contrast
    
    Args:
        model: Trained Keras model
        image_path: Path to input image
        output_path: Path to save visualization
        class_index: Optional class index
        predictions: Optional predictions array
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (96, 96))
        img_array = img_resized.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get class index if not provided
        if class_index is None and predictions is not None:
            class_index = np.argmax(predictions[0])
        
        # Generate saliency map
        saliency = SaliencyMap(model)
        saliency_map = saliency.generate_saliency(img_array, class_index)
        
        # Resize saliency to match original image
        saliency_resized = cv2.resize(saliency_map, (img_rgb.shape[1], img_rgb.shape[0]))
        
        # Create high-contrast visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img_rgb)
        axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Saliency map
        im = axes[1].imshow(saliency_resized, cmap='hot', interpolation='bilinear')
        axes[1].set_title('Saliency Map (Hot Colormap)', fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        
        # Overlay
        saliency_colored = plt.cm.hot(saliency_resized)[:, :, :3]  # Remove alpha
        overlay = img_rgb.astype(np.float32) / 255.0 * 0.4 + saliency_colored * 0.6
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay Visualization', fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close('all')
        
        print(f"[OK] Saliency map saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error creating saliency visualization: {e}")
        import traceback
        traceback.print_exc()
        raise


def create_grid_visualization(model, image_path, output_path, predictions=None):
    """
    Create a 2x2 grid showing each class separately with distinct colors
    This makes it very clear which regions are important for each class
    
    Args:
        model: Trained Keras model
        image_path: Path to input image
        output_path: Path to save visualization
        predictions: Optional predictions array
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (96, 96))
        img_array = img_resized.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get predictions
        if predictions is None:
            predictions = model.predict(img_array, verbose=0)
        
        # Generate heatmaps for all classes
        multicam = MultiClassGradCAM(model)
        heatmaps = multicam.generate_class_heatmaps(img_array, predictions)
        
        # Class colors and names
        colors = [
            (0, 255, 0),      # Green - NonDemented
            (255, 255, 0),    # Yellow - VeryMildDemented
            (255, 165, 0),   # Orange - MildDemented
            (255, 0, 0),     # Red - ModerateDemented
        ]
        class_names = ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented']
        
        # Create 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()
        
        for class_idx in range(4):
            heatmap = heatmaps[class_idx]
            heatmap_resized = cv2.resize(heatmap, (img_resized.shape[1], img_resized.shape[0]))
            
            # Normalize
            if np.max(heatmap_resized) > 0:
                heatmap_normalized = heatmap_resized / np.max(heatmap_resized)
            else:
                heatmap_normalized = heatmap_resized
            
            # Create colored heatmap
            color = np.array(colors[class_idx], dtype=np.float32) / 255.0
            colored_heatmap = np.zeros_like(img_resized, dtype=np.float32)
            for c in range(3):
                colored_heatmap[:, :, c] = heatmap_normalized * color[c]
            
            # Blend with original (high contrast)
            overlay = img_resized.astype(np.float32) / 255.0 * 0.3 + colored_heatmap * 0.7
            
            # Display
            axes[class_idx].imshow(overlay)
            prob = predictions[0][class_idx] * 100
            axes[class_idx].set_title(
                f'{class_names[class_idx]}\nProbability: {prob:.1f}%',
                fontsize=11, fontweight='bold',
                color=tuple(c/255.0 for c in colors[class_idx])
            )
            axes[class_idx].axis('off')
        
        plt.suptitle('Class-Specific Activation Maps', fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close('all')
        
        print(f"[OK] Grid visualization saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error creating grid visualization: {e}")
        import traceback
        traceback.print_exc()
        raise


def create_multi_class_gradcam(model, image_path, output_path, predictions=None):
    """
    Convenience function to create multi-class Grad-CAM visualization
    
    Args:
        model: Trained Keras model
        image_path: Path to input image
        output_path: Path to save visualization
        predictions: Optional predictions array
    
    Returns:
        Path to saved visualization
    """
    try:
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (96, 96))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Get predictions if not provided
        if predictions is None:
            _ = model.predict(img, verbose=0)  # Build model
            predictions = model.predict(img, verbose=0)
        
        # Create multi-class Grad-CAM
        multicam = MultiClassGradCAM(model)
        
        # Generate visualization
        multicam.create_multi_class_visualization(
            img,
            original_image_path=image_path,
            predictions=predictions,
            save_path=output_path
        )
        
        return output_path
    except Exception as e:
        print(f"Error in create_multi_class_gradcam: {e}")
        import traceback
        traceback.print_exc()
        raise


def create_gradcam(model, image_path, output_path, class_index=None):
    """
    Convenience function to create and save Grad-CAM visualization
    
    Args:
        model: Trained Keras model (must be already loaded and built)
        image_path: Path to input image
        output_path: Path to save visualization
        class_index: Optional class index for heatmap
    
    Returns:
        Path to saved visualization
    """
    try:
        # Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (96, 96))
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Ensure model is built by running a prediction first
        # This is critical for Sequential models
        _ = model.predict(img, verbose=0)
        
        # Convert to tensor for consistency
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
        
        # Create Grad-CAM (model should now be built from prediction above)
        # Pass the model which has been called, so outputs are available
        gradcam = GradCAM(model)
        
        # Generate visualization
        gradcam.generate_visualization(
            img_tensor, 
            original_image_path=image_path,
            class_index=class_index,
            save_path=output_path
        )
        
        return output_path
    except Exception as e:
        print(f"Error in create_gradcam: {e}")
        import traceback
        traceback.print_exc()
        raise

