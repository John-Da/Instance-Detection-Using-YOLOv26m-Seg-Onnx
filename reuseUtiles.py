import numpy as np
import onnxruntime as ort
import time
import os
import yaml
import cv2
import math
import matplotlib.pyplot as plt


class ReuseUtils:
    """
    A utility class for object detection and instance segmentation using ONNX models.
    Supports both detection (bounding boxes) and instance segmentation (masks).
    """
    
    def __init__(self, model_path, custom_labels=None, title="Detection Results", conf_thres=0.7, iou_thres=0.5, num_masks=32, input_size=640):
        """
        Initialize the ReuseUtils class.
        
        Args:
            model_path (str): Path to the ONNX model file
            custom_labels (str or list, optional): Custom labels (file path or list)
            title (str): Title for visualization
            conf_thres (float): Confidence threshold for detections
            iou_thres (float): IoU threshold for NMS
            num_masks (int): Number of mask coefficients
            input_size (int): Model input size (default: 640)
        """
        self.title = title
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.num_masks = num_masks
        self.input_size = input_size
        self.img_height = None
        self.img_width = None
        
        # Initialize model
        self.session = None
        self.load_model(model_path)
        
        # Load labels
        self.custom_labels = self.load_custom_labels(custom_labels)

    def load_model(self, model_path):
        """Load ONNX model with available providers."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            self.session = ort.InferenceSession(
                model_path, 
                providers=ort.get_available_providers()
            )
            print(f"Model loaded successfully: {model_path}")
            print(f"Available providers: {self.session.get_providers()}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def get_image_list(self, input_path, exts=(".jpg", ".jpeg", ".png")):

        # Case 1: list / tuple / set → always list
        if isinstance(input_path, (list, tuple, set)):
            images = []
            for item in input_path:
                result = self.get_image_list(item, exts)
                if isinstance(result, list):
                    images.extend(result)
                else:
                    images.append(result)
            return images

        # Case 2: single image file → return STRING
        if isinstance(input_path, str) and os.path.isfile(input_path):
            if input_path.lower().endswith(exts):
                return input_path
            else:
                raise ValueError(f"Not a supported image file: {input_path}")

        # Case 3: directory → return LIST
        if isinstance(input_path, str) and os.path.isdir(input_path):
            images = [
                os.path.join(input_path, f)
                for f in os.listdir(input_path)
                if f.lower().endswith(exts)
            ]

            if not images:
                raise ValueError(f"No images found in folder: {input_path}")

            return sorted(images)

        raise TypeError(
            "input_path must be an image path, folder path, or list of them"
        )
    

    def load_image(self, image_path):
        """
        Load and preprocess image for inference.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            np.ndarray: Preprocessed image tensor
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        self.img_height, self.img_width = image.shape[:2]
        
        # Resize and normalize
        img_resized = cv2.resize(image, (self.input_size, self.input_size))
        input_img = img_resized / 255.0
        input_img = input_img.transpose(2, 0, 1)  # HWC to CHW
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        
        return input_tensor, image

    def inference(self, input_tensor):
        """
        Run inference on the input tensor.
        
        Args:
            input_tensor (np.ndarray): Preprocessed image tensor
            
        Returns:
            list: Model outputs
        """
        start = time.perf_counter()
        outputs = self.session.run(
            None, 
            {self.session.get_inputs()[0].name: input_tensor}
        )
        inference_time = (time.perf_counter() - start) * 1000  # Convert to ms
        return outputs
    
    def start_detection(self, image_path, mode="det", save_img=False, save_dir="results"):
        """
        Perform detection or segmentation on images.
        
        Args:
            images (str or list): Image path(s)
            mode (str): "det" for detection, "inst" for instance segmentation
            save_img (bool): Whether to save results
            save_dir (str): Directory to save results
        """
        images = self.get_image_list(image_path)
        
        if mode not in ["det", "inst"]:
            raise ValueError(f"Unknown mode: {mode}. Use 'det' or 'inst'")
        
        try:
            if save_img:
                os.makedirs(save_dir, exist_ok=True)

            def process_single_image(image_path, idx=None):
                input_tensor, original_image = self.load_image(image_path)
                outputs = self.inference(input_tensor)
                detections = self.decode_detections(outputs[0][0])

                if mode == "det":
                    result = self.draw_boxes(original_image, detections)
                elif mode == "inst":
                    if len(outputs) < 2:
                        raise ValueError("Model output does not contain mask prototypes")
                    result = self.draw_masks(original_image, detections, outputs[1][0])

                if save_img:
                    name = f"{self.title}_{mode}"
                    if idx is not None:
                        name += f"_{idx:03d}"
                    save_path = os.path.join(save_dir, f"{name}.jpg")
                    cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
                    print(f"Saved: {save_path}")
                    
                return result

            # Handle single image or list of images
            if isinstance(images, list):
                rendered = [process_single_image(img, i) for i, img in enumerate(images)]
                self.visualize_images(rendered)
            else:
                result = process_single_image(images)
                self.visualize_images(result)

        except Exception as e:
            print(f"Detection error: {e}")
            raise

    def decode_detections(self, det_out):
        """
        Decode raw detection outputs.
        
        Args:
            det_out (np.ndarray): Raw detection output from model
            
        Returns:
            list: List of detection dictionaries
        """
        detections = []

        for det in det_out:
            obj_conf = det[4]
            if obj_conf >= self.conf_thres:
                class_id = int(det[5])
                mask_coeffs = det[6:6 + self.num_masks]

                detections.append({
                    "bbox": det[:4],
                    "score": obj_conf,
                    "class_id": class_id,
                    "mask": mask_coeffs
                })
        
        return detections

    def get_random_color(self, seed=5):
        """Generate random colors for visualization."""
        np.random.seed(seed)
        colors = np.random.randint(0, 255, size=(100, 3))
        return colors

    def sigmoid(self, x):
        """Apply sigmoid activation."""
        return 1 / (1 + np.exp(-x))

    def draw_boxes(self, image, detections):
        """
        Draw bounding boxes on the image.
        
        Args:
            image (np.ndarray): Original image
            detections (list): List of detections
            
        Returns:
            np.ndarray: Image with drawn boxes
        """
        h, w = image.shape[:2]
        sx, sy = w / self.input_size, h / self.input_size

        img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        random_colors = self.get_random_color()

        for det in detections:
            cx, cy, bw, bh = det["bbox"]
            class_id = det["class_id"]
            score = det["score"]

            x1 = int(cx * sx)
            y1 = int(cy * sy)
            x2 = int(bw * sx)
            y2 = int(bh * sy)

            color = tuple(int(c) for c in random_colors[class_id % len(random_colors)])
            label = f"{self.custom_labels[class_id]}: {score:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return img

    def draw_masks(self, image, detections, proto):
        """
        Draw instance segmentation masks on the image.
        
        Args:
            image (np.ndarray): Original image
            detections (list): List of detections
            proto (np.ndarray): Prototype masks from model
            
        Returns:
            np.ndarray: Image with drawn masks
        """
        h, w = image.shape[:2]
        sx, sy = w / self.input_size, h / self.input_size

        img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
        random_colors = self.get_random_color()

        # Reshape prototypes
        proto = proto.reshape(self.num_masks, 160 * 160)

        for det in detections:
            cx, cy, bw, bh = det["bbox"]
            class_id = det["class_id"]
            score = det["score"]

                        # Convert from center format to corner format
            x1 = int(cx * sx)
            y1 = int(cy * sy)
            x2 = int(bw * sx)
            y2 = int(bh * sy)

            # Generate mask
            mask_coeffs = det["mask"]
            mask = self.sigmoid(mask_coeffs @ proto)
            mask = mask.reshape(160, 160)

            # Resize mask to original image size
            mask = cv2.resize(mask, (w, h))
            mask_binary = (mask > 0.5).astype(np.uint8)

            # Crop mask to bbox
            mask_box = mask_binary[y1:y2, x1:x2]
            
            if mask_box.size == 0:
                continue

            color = random_colors[class_id % len(random_colors)].astype(np.float32)

            # Apply mask to region
            region = img[y1:y2, x1:x2].astype(np.float32)
            
            # Ensure dimensions match
            if mask_box.shape[0] == region.shape[0] and mask_box.shape[1] == region.shape[1]:
                img[y1:y2, x1:x2] = np.where(
                    mask_box[:, :, None],
                    region * 0.4 + color * 0.6,
                    region
                ).astype(np.uint8)

            # Draw bbox
            color_int = tuple(int(c) for c in color)
            cv2.rectangle(img, (x1, y1), (x2, y2), color_int, 2)
            
            label = f"{self.custom_labels[class_id]}: {score:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w, y1), color_int, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return img

    def visualize_images(self, images, cols=3):
        """
        Visualize images using matplotlib.
        
        Args:
            images (np.ndarray or list): Image(s) to visualize
            cols (int): Number of columns for grid layout
        """
        try:
            if isinstance(images, list):
                cols = min(len(images), cols)
                rows = math.ceil(len(images) / cols)
                fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
                axes = np.atleast_1d(axes).flatten()

                for i, img in enumerate(images):
                    axes[i].imshow(img)
                    axes[i].axis("off")

                # Hide unused subplots
                for j in range(len(images), len(axes)):
                    axes[j].axis("off")

                fig.suptitle(self.title, fontsize=16)
                plt.tight_layout()
                plt.show()
            else:
                plt.figure(figsize=(10, 8))
                plt.imshow(images)
                plt.axis("off")
                plt.title(self.title, fontsize=16)
                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(f"Visualization error: {e}")
    
    def load_custom_labels(self, custom_labels=None):
        """
        Load custom labels from file or use COCO labels.
        
        Args:
            custom_labels (str or list, optional): Path to label file or list of labels
            
        Returns:
            list: List of label names
        """
        coco_labels = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", 
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", 
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", 
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", 
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", 
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", 
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
            "toothbrush"
        ]

        # Return custom labels if provided as list
        if isinstance(custom_labels, list):
            return custom_labels

        # Load from file
        if isinstance(custom_labels, str):
            ext = os.path.splitext(custom_labels)[1].lower()
            
            if ext == ".txt":
                try:
                    with open(custom_labels, "r") as f:
                        labels = [line.strip() for line in f if line.strip()]
                    print(f"Loaded {len(labels)} labels from {custom_labels}")
                    return labels
                except FileNotFoundError:
                    raise FileNotFoundError(f"Label file not found: {custom_labels}")
                    
            elif ext in (".yaml", ".yml"):
                try:
                    with open(custom_labels, "r") as f:
                        data = yaml.safe_load(f)
                        labels = data.get("names", [])
                    print(f"Loaded {len(labels)} labels from {custom_labels}")
                    return labels
                except FileNotFoundError:
                    raise FileNotFoundError(f"Label file not found: {custom_labels}")
            else:
                raise ValueError(f"Unsupported file format: {ext}. Use .txt, .yaml, or .yml")

        # Use COCO labels by default
        print(f"Using default COCO labels ({len(coco_labels)} classes)")
        print("")
        return coco_labels
    
    def __str__(self):
        """Return string representation of the object."""
        return (
            f"Model: {self.session.get_inputs()[0].name if self.session else 'Not loaded'}\n"
            f"Input size: {self.input_size}x{self.input_size}\n"
            f"Confidence threshold: {self.conf_thres}\n"
            f"IoU threshold: {self.iou_thres}\n"
            f"Number of classes: {len(self.custom_labels)}\n"
            f"\nTitle: {self.title}\n"
            )
    
    def __repr__(self):
        """Return detailed representation."""
        return self.__str__()


# Example usage
# if __name__ == "__main__":
#     # Initialize the detector
#     detector = ReuseUtils(
#         model_path="yolov8n-seg.onnx",
#         title="YOLOv8 Detection",
#         conf_thres=0.5
#     )
    
#     # Run detection
#     detector.start_detection(
#         images="image.jpg",
#         mode="det",
#         save_img=True,
#         save_dir="results"
#     )