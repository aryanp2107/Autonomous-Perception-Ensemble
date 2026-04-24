"""
Autonomous Perception Ensemble - Inference Pipeline

Combines YOLOv8 (detection) + U-Net (segmentation) + MiDaS (depth)
into a unified perception system.
"""

import numpy as np
import cv2
from PIL import Image
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import onnxruntime as ort


@dataclass
class Detection:
    """Single detection result"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    depth: Optional[float] = None  # Estimated depth


@dataclass
class PerceptionResult:
    """Combined perception output"""
    detections: List[Detection]
    segmentation_mask: np.ndarray  # H x W, values: 0=background, 1=drivable
    depth_map: np.ndarray  # H x W, relative depth values
    fused_image: Optional[np.ndarray] = None


class PerceptionEnsemble:
    """
    Multi-model perception ensemble for autonomous driving.
    
    Combines:
    - YOLOv8 for object detection
    - U-Net for drivable area segmentation  
    - MiDaS for monocular depth estimation
    """
    
    # BDD100K class names
    CLASS_NAMES = [
        'car', 'truck', 'bus', 'person', 'rider',
        'bicycle', 'motorcycle', 'traffic light', 'traffic sign', 'train'
    ]
    
    def __init__(
        self,
        detection_model: str,
        segmentation_model: str,
        depth_model: str,
        device: str = 'cpu'
    ):
        """
        Initialize the perception ensemble.
        
        Args:
            detection_model: Path to YOLOv8 ONNX model
            segmentation_model: Path to U-Net ONNX model
            depth_model: Path to MiDaS ONNX model
            device: 'cpu' or 'cuda'
        """
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        
        print("Loading models...")
        self.detector = ort.InferenceSession(detection_model, providers=providers)
        self.segmentor = ort.InferenceSession(segmentation_model, providers=providers)
        self.depth_estimator = ort.InferenceSession(depth_model, providers=providers)
        print("Models loaded!")
        
        # Get input shapes
        self.det_input_shape = self.detector.get_inputs()[0].shape[2:4]  # H, W
        self.seg_input_shape = self.segmentor.get_inputs()[0].shape[2:4]
        self.depth_input_shape = self.depth_estimator.get_inputs()[0].shape[2:4]
    
    def preprocess_detection(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for YOLOv8"""
        h, w = self.det_input_shape
        img = cv2.resize(image, (w, h))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, 0)  # Add batch dim
        return img
    
    def preprocess_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for U-Net"""
        h, w = self.seg_input_shape
        img = cv2.resize(image, (w, h))
        img = img.astype(np.float32) / 255.0
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0).astype(np.float32)
        return img
    
    def preprocess_depth(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for MiDaS"""
        h, w = self.depth_input_shape
        img = cv2.resize(image, (w, h))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        return img
    
    def run_detection(self, image: np.ndarray, conf_threshold: float = 0.25) -> List[Detection]:
        """Run YOLOv8 detection"""
        orig_h, orig_w = image.shape[:2]
        input_tensor = self.preprocess_detection(image)
        
        # Run inference
        outputs = self.detector.run(None, {self.detector.get_inputs()[0].name: input_tensor})
        predictions = outputs[0]  # Shape: [1, num_detections, 5+num_classes]
        
        detections = []
        for pred in predictions[0]:
            # Parse YOLO output format
            x_center, y_center, width, height = pred[:4]
            obj_conf = pred[4] if len(pred) > 5 else 1.0
            class_scores = pred[5:] if len(pred) > 5 else pred[4:]
            
            class_id = np.argmax(class_scores)
            confidence = float(class_scores[class_id] * obj_conf)
            
            if confidence < conf_threshold:
                continue
            
            # Convert to pixel coordinates
            x1 = int((x_center - width / 2) * orig_w / self.det_input_shape[1])
            y1 = int((y_center - height / 2) * orig_h / self.det_input_shape[0])
            x2 = int((x_center + width / 2) * orig_w / self.det_input_shape[1])
            y2 = int((y_center + height / 2) * orig_h / self.det_input_shape[0])
            
            class_name = self.CLASS_NAMES[class_id] if class_id < len(self.CLASS_NAMES) else f"class_{class_id}"
            
            detections.append(Detection(
                class_id=int(class_id),
                class_name=class_name,
                confidence=confidence,
                bbox=(x1, y1, x2, y2)
            ))
        
        return detections
    
    def run_segmentation(self, image: np.ndarray) -> np.ndarray:
        """Run U-Net segmentation"""
        orig_h, orig_w = image.shape[:2]
        input_tensor = self.preprocess_segmentation(image)
        
        # Run inference
        outputs = self.segmentor.run(None, {self.segmentor.get_inputs()[0].name: input_tensor})
        mask = outputs[0][0, 0]  # Remove batch and channel dims
        
        # Sigmoid + threshold
        mask = 1 / (1 + np.exp(-mask))
        mask = (mask > 0.5).astype(np.uint8)
        
        # Resize to original
        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        return mask
    
    def run_depth(self, image: np.ndarray) -> np.ndarray:
        """Run MiDaS depth estimation"""
        orig_h, orig_w = image.shape[:2]
        input_tensor = self.preprocess_depth(image)
        
        # Run inference
        outputs = self.depth_estimator.run(None, {self.depth_estimator.get_inputs()[0].name: input_tensor})
        depth = outputs[0][0]
        
        # Normalize to 0-1
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        
        # Resize to original
        depth = cv2.resize(depth, (orig_w, orig_h))
        
        return depth
    
    def predict(self, image_path: str, conf_threshold: float = 0.25) -> PerceptionResult:
        """
        Run full perception pipeline on an image.
        
        Args:
            image_path: Path to input image
            conf_threshold: Detection confidence threshold
            
        Returns:
            PerceptionResult with detections, segmentation, and depth
        """
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run all models
        detections = self.run_detection(image, conf_threshold)
        segmentation = self.run_segmentation(image)
        depth = self.run_depth(image)
        
        # Add depth to detections
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            # Get median depth in bbox
            bbox_depth = depth[y1:y2, x1:x2]
            if bbox_depth.size > 0:
                det.depth = float(np.median(bbox_depth))
        
        return PerceptionResult(
            detections=detections,
            segmentation_mask=segmentation,
            depth_map=depth
        )
    
    def visualize(
        self,
        result: PerceptionResult,
        original_image: np.ndarray,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Create visualization of perception results.
        
        Args:
            result: PerceptionResult from predict()
            original_image: Original RGB image
            save_path: Optional path to save visualization
            
        Returns:
            Fused visualization image
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Original with detections
        det_img = original_image.copy()
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(det_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            if det.depth is not None:
                label += f" ({det.depth:.1f}m)"
            cv2.putText(det_img, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        axes[0, 0].imshow(det_img)
        axes[0, 0].set_title(f'Detection ({len(result.detections)} objects)')
        axes[0, 0].axis('off')
        
        # Segmentation
        axes[0, 1].imshow(original_image)
        axes[0, 1].imshow(result.segmentation_mask, alpha=0.5, cmap='Greens')
        axes[0, 1].set_title('Drivable Area')
        axes[0, 1].axis('off')
        
        # Depth
        axes[1, 0].imshow(result.depth_map, cmap='plasma')
        axes[1, 0].set_title('Depth Estimation')
        axes[1, 0].axis('off')
        
        # Fused
        fused = original_image.copy().astype(np.float32)
        # Add green tint to drivable area
        fused[result.segmentation_mask == 1, 1] += 50
        fused = np.clip(fused, 0, 255).astype(np.uint8)
        # Draw detections
        for det in result.detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(fused, (x1, y1), (x2, y2), (255, 0, 0), 2)
        axes[1, 1].imshow(fused)
        axes[1, 1].set_title('Fused Perception')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
        
        return fused


if __name__ == "__main__":
    # Example usage
    ensemble = PerceptionEnsemble(
        detection_model="models/yolov8n_bdd100k.onnx",
        segmentation_model="models/unet_drivable.onnx",
        depth_model="models/midas_small.onnx"
    )
    
    result = ensemble.predict("test_image.jpg")
    
    print(f"Detections: {len(result.detections)}")
    for det in result.detections:
        print(f"  {det.class_name}: {det.confidence:.2f} @ {det.bbox}")
    
    print(f"Drivable area: {result.segmentation_mask.mean()*100:.1f}% of image")
