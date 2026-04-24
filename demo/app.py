"""
Autonomous Perception Ensemble - Gradio Demo

Interactive demo for the perception pipeline.
Upload a dashcam image → Get detection + segmentation + depth
"""

import gradio as gr
import numpy as np
import cv2
from PIL import Image
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.inference import PerceptionEnsemble, PerceptionResult


# Global model (loaded once)
ensemble = None


def load_models():
    """Load ONNX models"""
    global ensemble
    
    if ensemble is not None:
        return ensemble
    
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    ensemble = PerceptionEnsemble(
        detection_model=os.path.join(model_dir, "yolov8n_bdd100k.onnx"),
        segmentation_model=os.path.join(model_dir, "unet_drivable.onnx"),
        depth_model=os.path.join(model_dir, "midas_small.onnx")
    )
    
    return ensemble


def create_visualization(
    image: np.ndarray,
    result: PerceptionResult,
    show_detections: bool,
    show_segmentation: bool,
    show_depth: bool
) -> np.ndarray:
    """Create custom visualization based on toggles"""
    
    output = image.copy().astype(np.float32)
    
    # Segmentation overlay
    if show_segmentation:
        mask_overlay = np.zeros_like(output)
        mask_overlay[result.segmentation_mask == 1] = [0, 255, 0]  # Green
        output = cv2.addWeighted(output, 0.7, mask_overlay, 0.3, 0)
    
    # Depth overlay (optional blend)
    if show_depth and not show_segmentation:
        depth_colored = cv2.applyColorMap(
            (result.depth_map * 255).astype(np.uint8),
            cv2.COLORMAP_PLASMA
        )
        depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)
        output = cv2.addWeighted(output.astype(np.uint8), 0.6, depth_colored, 0.4, 0)
    
    output = output.astype(np.uint8)
    
    # Detection boxes
    if show_detections:
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            
            # Color based on class
            colors = {
                'car': (255, 100, 100),
                'truck': (255, 150, 50),
                'bus': (255, 200, 50),
                'person': (100, 255, 100),
                'rider': (100, 200, 255),
                'bicycle': (200, 100, 255),
                'motorcycle': (255, 100, 200),
            }
            color = colors.get(det.class_name, (200, 200, 200))
            
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"{det.class_name} {det.confidence:.0%}"
            if det.depth is not None:
                # Convert relative depth to approximate meters (rough estimate)
                approx_dist = (1 - det.depth) * 50  # Closer = higher depth value
                label += f" ~{approx_dist:.0f}m"
            
            # Background for text
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(output, label, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return output


def predict(
    image: Image.Image,
    confidence: float,
    show_detections: bool,
    show_segmentation: bool,
    show_depth: bool
):
    """Main prediction function for Gradio"""
    
    if image is None:
        return None, "Please upload an image"
    
    # Load models
    model = load_models()
    
    # Convert to numpy
    img_np = np.array(image)
    
    # Run perception
    result = model.predict_array(img_np, conf_threshold=confidence)
    
    # Create visualization
    output = create_visualization(
        img_np, result,
        show_detections, show_segmentation, show_depth
    )
    
    # Create summary
    summary_lines = [
        f"**Detections:** {len(result.detections)} objects",
        "",
        "| Object | Confidence | Distance |",
        "|--------|------------|----------|"
    ]
    
    for det in sorted(result.detections, key=lambda x: -x.confidence):
        dist = f"~{(1-det.depth)*50:.0f}m" if det.depth else "N/A"
        summary_lines.append(f"| {det.class_name} | {det.confidence:.0%} | {dist} |")
    
    summary_lines.extend([
        "",
        f"**Drivable Area:** {result.segmentation_mask.mean()*100:.1f}% of image"
    ])
    
    summary = "\n".join(summary_lines)
    
    return output, summary


# Extend PerceptionEnsemble for array input
def predict_array(self, image: np.ndarray, conf_threshold: float = 0.25):
    """Run prediction on numpy array directly"""
    detections = self.run_detection(image, conf_threshold)
    segmentation = self.run_segmentation(image)
    depth = self.run_depth(image)
    
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        bbox_depth = depth[y1:y2, x1:x2]
        if bbox_depth.size > 0:
            det.depth = float(np.median(bbox_depth))
    
    from src.inference import PerceptionResult
    return PerceptionResult(
        detections=detections,
        segmentation_mask=segmentation,
        depth_map=depth
    )

# Monkey-patch
PerceptionEnsemble.predict_array = predict_array


# Build Gradio interface
with gr.Blocks(title="Autonomous Perception Ensemble", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🚗 Autonomous Perception Ensemble
    
    Multi-model sensor fusion for scene understanding.
    
    **Models:** YOLOv8 (detection) + U-Net (segmentation) + MiDaS (depth)
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Dashcam Image")
            
            confidence = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.25, step=0.05,
                label="Detection Confidence Threshold"
            )
            
            with gr.Row():
                show_det = gr.Checkbox(value=True, label="Detections")
                show_seg = gr.Checkbox(value=True, label="Drivable Area")
                show_depth = gr.Checkbox(value=False, label="Depth Map")
            
            run_btn = gr.Button("🔍 Analyze", variant="primary")
        
        with gr.Column(scale=1):
            output_image = gr.Image(label="Perception Output")
            output_summary = gr.Markdown(label="Summary")
    
    # Examples
    gr.Examples(
        examples=[
            ["assets/example1.jpg", 0.25, True, True, False],
            ["assets/example2.jpg", 0.3, True, False, True],
        ],
        inputs=[input_image, confidence, show_det, show_seg, show_depth],
        outputs=[output_image, output_summary],
        fn=predict,
        cache_examples=False
    )
    
    # Event handlers
    run_btn.click(
        fn=predict,
        inputs=[input_image, confidence, show_det, show_seg, show_depth],
        outputs=[output_image, output_summary]
    )
    
    gr.Markdown("""
    ---
    **Built by [Aryan Patel](https://github.com/aryanp2107)** | 
    Part of 100 ML Projects Challenge
    """)


if __name__ == "__main__":
    demo.launch(share=True)
