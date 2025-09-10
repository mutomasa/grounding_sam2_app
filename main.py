import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import pandas as pd
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import supervision as sv
from ultralytics import YOLO
import random

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    st.warning("⚠️ Segment Anything (SAM v1) not available. Using simplified segmentation if SAM2 also missing.")

# Optional: SAM2 (Segment Anything 2)
SAM2_IMPORT_ERROR = None
try:
    # SAM2 official package imports
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except Exception as e:
    SAM2_AVAILABLE = False
    SAM2_IMPORT_ERROR = str(e)

# Configure page
st.set_page_config(
    page_title="Grounding SAM 2 App",
    page_icon="🎯",
    layout="wide"
)

class GroundingSAM2Pipeline:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.grounding_dino_processor = None
        self.grounding_dino_model = None
        self.sam_predictor = None  # SAM v1
        self.sam2_predictor = None  # SAM v2
        self.yolo_model = None  # Fallback
        self.load_models()
    
    def load_models(self):
        """Load GroundingDINO and SAM models"""
        models_loaded = []
        
        try:
            # 1. Load Grounding DINO from HuggingFace
            st.info("🔄 Loading Grounding DINO model...")
            model_id = "IDEA-Research/grounding-dino-tiny"
            self.grounding_dino_processor = AutoProcessor.from_pretrained(model_id)
            self.grounding_dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
            models_loaded.append("✅ Grounding DINO")
            
        except Exception as e:
            st.warning(f"⚠️ Failed to load Grounding DINO: {e}")
            models_loaded.append("❌ Grounding DINO")
        
        # 2. Try SAM2 first, then SAM v1
        try:
            if SAM2_AVAILABLE:
                st.info("🔄 Loading SAM2 predictor (if checkpoints exist)...")
                ckpt_dir = Path("checkpoints")
                ckpt_dir.mkdir(exist_ok=True)
                # Try to locate typical SAM2 checkpoint files in checkpoints directory
                sam2_ckpt = None
                for p in ckpt_dir.glob("**/*"):
                    name = p.name.lower()
                    if p.is_file() and name.endswith((".pt", ".pth")) and "sam2" in name:
                        sam2_ckpt = str(p)
                        break
                
                if sam2_ckpt:
                    # Use built-in config name instead of local config file
                    config_name = "sam2_hiera_b+.yaml"  # Use the built-in config
                    sam2_model = build_sam2(config_name, sam2_ckpt, device=self.device)
                    self.sam2_predictor = SAM2ImagePredictor(sam2_model)
                    models_loaded.append("✅ SAM2")
                else:
                    models_loaded.append("❌ SAM2 (checkpoint not found)")
            else:
                models_loaded.append("❌ SAM2 (not installed)")
        except Exception as e:
            st.warning(f"⚠️ Failed to load SAM2: {e}")
            models_loaded.append("❌ SAM2")

        try:
            # Fallback to SAM v1 if SAM2 not loaded
            if self.sam2_predictor is None and SAM_AVAILABLE:
                st.info("🔄 Loading SAM v1 predictor...")
                sam_checkpoint_path = self.download_sam_checkpoint()
                if sam_checkpoint_path:
                    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint_path)
                    sam.to(device=self.device)
                    self.sam_predictor = SamPredictor(sam)
                    models_loaded.append("✅ SAM v1")
                else:
                    models_loaded.append("❌ SAM v1 (checkpoint not found)")
            elif self.sam2_predictor is None and not SAM_AVAILABLE:
                models_loaded.append("❌ SAM v1 (not installed)")
        except Exception as e:
            st.warning(f"⚠️ Failed to load SAM v1: {e}")
            models_loaded.append("❌ SAM v1")
        
        try:
            # 3. Load YOLO as fallback
            st.info("🔄 Loading YOLO fallback...")
            self.yolo_model = YOLO('yolov8n.pt')
            models_loaded.append("✅ YOLO (fallback)")
            
        except Exception as e:
            st.error(f"❌ Failed to load YOLO fallback: {e}")
            models_loaded.append("❌ YOLO (fallback)")
        
        # Display loaded models
        st.success("Models loaded:")
        for model_status in models_loaded:
            st.write(f"  {model_status}")
            
        # Determine primary detection method
        if self.grounding_dino_model is not None:
            st.info("🎯 Primary detection: Grounding DINO")
            self.detection_method = "grounding_dino"
        elif self.yolo_model is not None:
            st.info("🎯 Primary detection: YOLO (fallback)")
            self.detection_method = "yolo"
        else:
            st.error("❌ No object detection model available!")
            self.detection_method = None
    
    def download_sam_checkpoint(self):
        """Download SAM checkpoint if not available"""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint_path = checkpoint_dir / "sam_vit_h_4b8939.pth"
        
        if checkpoint_path.exists():
            return str(checkpoint_path)
        
        try:
            import urllib.request
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            st.info(f"📥 Downloading SAM checkpoint ({url})...")
            
            # Show progress bar
            progress_bar = st.progress(0)
            
            def progress_hook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(downloaded / total_size, 1.0)
                progress_bar.progress(percent)
            
            urllib.request.urlretrieve(url, checkpoint_path, progress_hook)
            progress_bar.empty()
            
            st.success("✅ SAM checkpoint downloaded successfully!")
            return str(checkpoint_path)
            
        except Exception as e:
            st.warning(f"⚠️ Failed to download SAM checkpoint: {e}")
            return None
    
    def detect_objects_in_frame(self, frame, text_prompt):
        """Detect objects in a single frame based on text prompt"""
        if self.detection_method == "grounding_dino":
            return self._detect_with_grounding_dino(frame, text_prompt)
        elif self.detection_method == "yolo":
            return self._detect_with_yolo(frame, text_prompt)
        else:
            st.error("❌ No detection method available")
            return [], []
    
    def _post_process_gdino(self, outputs, input_ids, image_size, box_threshold, text_threshold):
        """Robust post-process for GroundingDINO across transformers versions."""
        target_sizes = [image_size[::-1]]  # (H, W)
        try:
            return self.grounding_dino_processor.post_process_grounded_object_detection(
                outputs, input_ids, box_threshold, text_threshold, target_sizes
            )[0]
        except TypeError:
            try:
                return self.grounding_dino_processor.post_process_grounded_object_detection(
                    outputs, input_ids, box_threshold, text_threshold
                )[0]
            except TypeError:
                try:
                    return self.grounding_dino_processor.post_process_grounded_object_detection(
                        outputs, input_ids, target_sizes=target_sizes
                    )[0]
                except Exception:
                    # Generic fallback
                    return self.grounding_dino_processor.post_process_object_detection(
                        outputs, target_sizes= torch.tensor(target_sizes), threshold=box_threshold
                    )[0]

    def _detect_with_grounding_dino(self, frame, text_prompt):
        """Detect objects using Grounding DINO"""
        try:
            # Convert frame to PIL Image
            if isinstance(frame, np.ndarray):
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
            else:
                image = frame
            
            # Prepare text prompt (must end with period for Grounding DINO)
            if not text_prompt.endswith('.'):
                text_prompt = text_prompt + '.'
            
            # Process inputs
            inputs = self.grounding_dino_processor(
                images=image, 
                text=text_prompt, 
                return_tensors="pt"
            ).to(self.device)
            
            # Get detections
            with torch.no_grad():
                outputs = self.grounding_dino_model(**inputs)
            
            # Get dynamic thresholds from session state
            box_threshold = getattr(st.session_state, 'box_threshold', 0.25)
            text_threshold = getattr(st.session_state, 'text_threshold', 0.25)
            
            # Post-process results (robust across versions)
            results = self._post_process_gdino(
                outputs, inputs.input_ids, image.size, box_threshold, text_threshold
            )
            
            detections = []
            all_detections = []
            
            if len(results["boxes"]) > 0:
                boxes = results["boxes"].cpu().numpy()
                scores = results["scores"].cpu().numpy()
                labels = results["labels"]
                
                for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                    x1, y1, x2, y2 = box
                    
                    detection_data = {
                        'class': label,
                        'confidence': float(score),
                        'bbox': [float(x1), float(y1), float(x2), float(y2)]
                    }
                    
                    all_detections.append(detection_data)
                    
                    # Get settings from session state
                    show_all = getattr(st.session_state, 'show_all_detections', False)
                    conf_threshold = getattr(st.session_state, 'confidence_threshold', 0.25)
                    
                    if show_all or score > conf_threshold:
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(score),
                            'class': label
                        })
            
            # Debug information
            if len(all_detections) > 0:
                debug_msg = f"Grounding DINO: Found {len(all_detections)} objects for '{text_prompt}': "
                debug_msg += ", ".join([f"{d['class']}({d['confidence']:.2f})" for d in all_detections[:5]])
                if len(all_detections) > 5:
                    debug_msg += f" and {len(all_detections)-5} more..."
                debug_msg += f" | Selected {len(detections)}"
                
                if hasattr(st.session_state, 'debug_mode') and st.session_state.get('debug_mode', False):
                    st.sidebar.text(debug_msg)
            
            return detections, all_detections
            
        except Exception as e:
            st.error(f"Error in Grounding DINO detection: {e}")
            # Fallback to YOLO if available
            if self.yolo_model is not None:
                st.warning("🔄 Falling back to YOLO detection...")
                return self._detect_with_yolo(frame, text_prompt)
            return [], []
    
    def _detect_with_yolo(self, frame, text_prompt):
        """Detect objects using YOLO (fallback method)"""
        try:
            results = self.yolo_model(frame)
            
            detections = []
            all_detections = []
            
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        class_name = self.yolo_model.names[cls]
                        
                        all_detections.append({
                            'class': class_name,
                            'confidence': float(conf),
                            'bbox': [float(x1), float(y1), float(x2), float(y2)]
                        })
                        
                        # Get settings from session state
                        show_all = getattr(st.session_state, 'show_all_detections', False)
                        conf_threshold = getattr(st.session_state, 'confidence_threshold', 0.3)
                        
                        should_include = False
                        if show_all and conf > conf_threshold:
                            should_include = True
                        elif (text_prompt.lower() in class_name.lower() or 
                              class_name.lower() in text_prompt.lower()):
                            should_include = True
                        elif conf > 0.7:
                            should_include = True
                        
                        if should_include:
                            detections.append({
                                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                'confidence': float(conf),
                                'class': class_name
                            })
            
            # Debug information
            if len(all_detections) > 0:
                debug_msg = f"YOLO: Found {len(all_detections)} total objects: "
                debug_msg += ", ".join([f"{d['class']}({d['confidence']:.2f})" for d in all_detections[:5]])
                if len(all_detections) > 5:
                    debug_msg += f" and {len(all_detections)-5} more..."
                debug_msg += f" | Matched {len(detections)} for prompt '{text_prompt}'"
                
                if hasattr(st.session_state, 'debug_mode') and st.session_state.get('debug_mode', False):
                    st.sidebar.text(debug_msg)
            
            return detections, all_detections
            
        except Exception as e:
            st.error(f"Error in YOLO detection: {e}")
            return [], []
    
    def generate_segmentation_mask(self, frame, bbox):
        """Generate segmentation mask using SAM or fallback method"""
        if self.sam2_predictor is not None:
            return self._generate_sam2_mask(frame, bbox)
        if self.sam_predictor is not None:
            return self._generate_sam_mask(frame, bbox)
        else:
            return self._generate_simple_mask(frame, bbox)
    
    def _generate_sam_mask(self, frame, bbox):
        """Generate segmentation mask using SAM"""
        try:
            # Convert frame to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            
            # Set image for SAM
            self.sam_predictor.set_image(frame_rgb)
            
            # Convert bbox to numpy array
            x1, y1, x2, y2 = map(int, bbox)
            input_box = np.array([x1, y1, x2, y2])
            
            # Generate mask using SAM
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=False,
            )
            
            # Return the best mask (first one when multimask_output=False)
            mask = masks[0].astype(np.uint8) * 255
            return mask
            
        except Exception as e:
            st.warning(f"⚠️ SAM segmentation failed: {e}")
            return self._generate_simple_mask(frame, bbox)

    def _generate_sam2_mask(self, frame, bbox):
        """Generate segmentation mask using SAM2 predictor."""
        try:
            # Convert BGR to RGB if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = frame
            self.sam2_predictor.set_image(frame_rgb)
            x1, y1, x2, y2 = map(int, bbox)
            input_box = np.array([x1, y1, x2, y2])[None, :]
            # SAM2 API: some versions use predict with box parameter name variations
            try:
                masks, _, _ = self.sam2_predictor.predict(box=input_box)
            except TypeError:
                masks, _, _ = self.sam2_predictor.predict(bboxes=input_box)
            # Assume first mask best
            mask = masks[0].astype(np.uint8) * 255
            return mask
        except Exception as e:
            st.warning(f"⚠️ SAM2 segmentation failed: {e}")
            # Fallback to SAM v1 or simple
            if self.sam_predictor is not None:
                return self._generate_sam_mask(frame, bbox)
            return self._generate_simple_mask(frame, bbox)
    
    def _generate_simple_mask(self, frame, bbox):
        """Generate a simple elliptical mask as fallback"""
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        x1, y1, x2, y2 = map(int, bbox)
        
        # Create an elliptical mask within the bounding box
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        width = x2 - x1
        height = y2 - y1
        
        cv2.ellipse(mask, (center_x, center_y), (width//2, height//2), 0, 0, 360, 255, -1)
        
        return mask
    
    def draw_detection_on_frame(self, frame, detection, color=None, viz_settings=None):
        """Draw detection results on frame with segmentation mask"""
        if color is None:
            # Generate bright, visible colors
            color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        
        if viz_settings is None:
            viz_settings = {
                'show_masks': True,
                'show_boxes': True,
                'show_labels': True,
                'mask_opacity': 0.3
            }
        
        # Extract detection info
        bbox = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class']
        
        # Ensure coordinates are valid
        x1, y1, x2, y2 = map(int, bbox)
        height, width = frame.shape[:2]
        
        # Clamp coordinates to frame bounds
        x1 = max(0, min(x1, width-1))
        y1 = max(0, min(y1, height-1))
        x2 = max(x1+1, min(x2, width-1))
        y2 = max(y1+1, min(y2, height-1))
        
        # Draw segmentation mask first (behind everything else)
        if viz_settings.get('show_masks', True):
            mask = self.generate_segmentation_mask(frame, [x1, y1, x2, y2])
            
            # Create colored overlay for the mask
            overlay = frame.copy()
            mask_indices = mask > 0
            
            if np.any(mask_indices):
                # Apply color to mask area
                for i, c in enumerate(color):
                    overlay[mask_indices, i] = c * 0.6 + overlay[mask_indices, i] * 0.4
                
                # Apply mask overlay with custom opacity
                mask_alpha = viz_settings.get('mask_opacity', 0.3)
                frame = cv2.addWeighted(frame, 1-mask_alpha, overlay, mask_alpha, 0)
        
        # Draw bounding box with thick, visible lines
        if viz_settings.get('show_boxes', True):
            # Draw thick rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            # Add inner border for better visibility
            inner_color = tuple(max(0, c-50) for c in color)
            cv2.rectangle(frame, (x1+1, y1+1), (x2-1, y2-1), inner_color, 1)
        
        # Draw label with background
        if viz_settings.get('show_labels', True):
            label = f"{class_name}: {confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            
            # Get label dimensions
            (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Ensure label fits in frame
            label_y = max(label_height + 10, y1)
            label_x1 = x1
            label_x2 = min(x1 + label_width + 10, width)
            label_y1 = label_y - label_height - 5
            label_y2 = label_y + 5
            
            # Draw label background
            cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, -1)
            
            # Draw label text in white
            cv2.putText(frame, label, (label_x1 + 5, label_y - 5), 
                       font, font_scale, (255, 255, 255), thickness)
        
        return frame
    
    def process_video(self, video_path, text_prompt):
        """Process video and track objects"""
        # Validate video path
        if not os.path.exists(video_path):
            st.error(f"Video file does not exist: {video_path}")
            return []
        
        # Check file size
        file_size = os.path.getsize(video_path)
        if file_size == 0:
            st.error("Video file is empty")
            return []
        
        st.info(f"Opening video file: {os.path.basename(video_path)} ({file_size / (1024*1024):.2f} MB)")
        
        # Try to open video with error handling
        cap = None
        try:
            # Use absolute path to avoid OpenCV path issues
            abs_video_path = os.path.abspath(video_path)
            
            # Try different backends in order
            backends = [cv2.CAP_ANY, cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER]
            
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(abs_video_path, backend)
                    if cap.isOpened():
                        # Test if we can read at least one frame
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            # Reset to beginning
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            break
                        else:
                            cap.release()
                            cap = None
                    else:
                        if cap:
                            cap.release()
                        cap = None
                except Exception as e:
                    if cap:
                        cap.release()
                    cap = None
                    continue
            
            if cap is None or not cap.isOpened():
                st.error(f"Could not open video file with any backend: {video_path}")
                return []
            
            # Get video properties for verification
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Validate video properties
            if width <= 0 or height <= 0:
                st.error(f"Invalid video dimensions: {width}x{height}")
                return []
                
            if fps <= 0:
                fps = 30.0  # Default fallback
                st.warning("Could not determine FPS, using default 30 FPS")
            
            st.success(f"Video opened successfully: {width}x{height}, {fps:.1f} FPS, {frame_count} frames")
            
        except Exception as e:
            st.error(f"Error opening video file: {str(e)}")
            if cap:
                cap.release()
            return []
        
        frames_data = []
        frame_idx = 0
        max_frames = min(100, frame_count) if frame_count > 0 else 100
        
        # Create progress bar
        progress_bar = st.progress(0)
        
        try:
            consecutive_failures = 0
            max_consecutive_failures = 10
            
            while frame_idx < max_frames and consecutive_failures < max_consecutive_failures:
                try:
                    # Attempt to read frame with additional error handling
                    try:
                        ret, frame = cap.read()
                    except Exception as read_error:
                        st.warning(f"Frame read error at {frame_idx}: {str(read_error)}")
                        consecutive_failures += 1
                        frame_idx += 1
                        continue
                    
                    if not ret:
                        st.info(f"End of video reached at frame {frame_idx}")
                        break
                    
                    # Validate frame with more thorough checks
                    if frame is None:
                        st.warning(f"Null frame at index {frame_idx}")
                        consecutive_failures += 1
                        frame_idx += 1
                        continue
                    
                    if frame.size == 0 or len(frame.shape) != 3:
                        st.warning(f"Invalid frame dimensions at index {frame_idx}: {frame.shape if frame is not None else 'None'}")
                        consecutive_failures += 1
                        frame_idx += 1
                        continue
                    
                    # Reset failure counter on successful frame read
                    consecutive_failures = 0
                    
                    # Detect objects in current frame
                    try:
                        detections, all_detections = self.detect_objects_in_frame(frame, text_prompt)
                        
                    except Exception as detection_error:
                        st.warning(f"Detection failed at frame {frame_idx}: {str(detection_error)}")
                        detections = []
                        all_detections = []
                    
                    frames_data.append({
                        'frame_idx': frame_idx,
                        'detections_count': len(detections),
                        'detections': detections,
                        'all_detections': all_detections,  # Store all detections for debugging
                        'frame': frame.copy()  # Store original frame
                    })
                    
                    frame_idx += 1
                    
                    # Update progress
                    progress = min(frame_idx / max_frames, 1.0)
                    progress_bar.progress(progress)
                    
                except Exception as frame_error:
                    st.warning(f"Frame processing error at {frame_idx}: {str(frame_error)}")
                    consecutive_failures += 1
                    frame_idx += 1
                    continue
            
            if consecutive_failures >= max_consecutive_failures:
                st.error(f"Too many consecutive frame reading failures. Stopping at frame {frame_idx}")
        
        except Exception as e:
            st.error(f"Critical error processing video at frame {frame_idx}: {str(e)}")
        
        finally:
            if cap:
                cap.release()
            progress_bar.empty()
        
        return frames_data
    
    def create_annotated_video(self, frames_data, output_path, fps=30, viz_settings=None):
        """Create video with annotations and segmentation masks"""
        if not frames_data:
            st.error("No frame data available for video creation")
            return None
        
        try:
            # Get video dimensions from first frame
            first_frame = frames_data[0]['frame']
            if first_frame is None or first_frame.size == 0:
                st.error("First frame is invalid")
                return None
                
            height, width = first_frame.shape[:2]
            st.info(f"Creating video with dimensions: {width}x{height}")
            
            # Ensure dimensions are even (required for some codecs)
            if height % 2 != 0:
                height -= 1
            if width % 2 != 0:
                width -= 1
            
            # Ensure minimum dimensions
            if width < 64 or height < 64:
                st.error(f"Video dimensions too small: {width}x{height}")
                return None
            
            # Try multiple codec options in order of preference (web-compatible first)
            codec_options = [
                ('H264', cv2.VideoWriter_fourcc(*'H264')),  # Best for web browsers
                ('avc1', cv2.VideoWriter_fourcc(*'avc1')),  # Alternative H.264
                ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),  # Fallback MP4
                ('XVID', cv2.VideoWriter_fourcc(*'XVID')),  # Fallback MPEG-4
                ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),  # Last resort
            ]
            
            out = None
            used_codec = None
            
            for codec_name, fourcc in codec_options:
                try:
                    temp_path = output_path.replace('.mp4', f'_{codec_name}.mp4')
                    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
                    
                    if out.isOpened():
                        used_codec = codec_name
                        output_path = temp_path
                        st.success(f"Using codec: {codec_name}")
                        break
                    else:
                        out.release()
                        out = None
                        
                except Exception as e:
                    st.warning(f"Codec {codec_name} failed: {str(e)}")
                    if out:
                        out.release()
                        out = None
                    continue
            
            if out is None or not out.isOpened():
                st.error("All video codecs failed. Cannot create video.")
                return None
            
        except Exception as e:
            st.error(f"Error initializing video writer: {str(e)}")
            return None
        
        # Define bright, distinguishable colors for different classes
        predefined_colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue  
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (0, 128, 255),  # Light Blue
            (255, 192, 203) # Pink
        ]
        
        class_colors = {}
        color_index = 0
        
        # Create progress bar for video creation
        video_progress = st.progress(0)
        
        try:
            for i, frame_data in enumerate(frames_data):
                try:
                    frame = frame_data['frame'].copy()
                    if frame is None or frame.size == 0:
                        st.warning(f"Skipping invalid frame at index {i}")
                        continue
                        
                    detections = frame_data['detections']
                    
                    # Resize frame if dimensions were adjusted
                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height))
                    
                    # Validate frame after resize
                    if frame.shape[:2] != (height, width):
                        st.warning(f"Frame resize failed at index {i}")
                        continue
                    
                    # Draw each detection
                    for detection in detections:
                        try:
                            class_name = detection['class']
                            
                            # Assign consistent bright color to class
                            if class_name not in class_colors:
                                if color_index < len(predefined_colors):
                                    class_colors[class_name] = predefined_colors[color_index]
                                else:
                                    # Fallback to random bright colors
                                    class_colors[class_name] = (
                                        random.randint(100, 255),
                                        random.randint(100, 255), 
                                        random.randint(100, 255)
                                    )
                                color_index += 1
                            
                            color = class_colors[class_name]
                            frame = self.draw_detection_on_frame(frame, detection, color, viz_settings)
                            
                        except Exception as e:
                            st.warning(f"Error drawing detection at frame {i}: {str(e)}")
                            continue
                    
                    # Add frame info with better visibility
                    try:
                        info_text = f"Frame: {frame_data['frame_idx']}, Detections: {frame_data['detections_count']}"
                        
                        # Add black background for text
                        text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        cv2.rectangle(frame, (5, 5), (text_size[0] + 15, 35), (0, 0, 0), -1)
                        cv2.putText(frame, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Add detection classes info
                        if detections:
                            class_info = f"Classes: {', '.join(set(d['class'] for d in detections))}"
                            class_text_size = cv2.getTextSize(class_info, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(frame, (5, 40), (class_text_size[0] + 15, 65), (0, 0, 0), -1)
                            cv2.putText(frame, class_info, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    except Exception as e:
                        st.warning(f"Error adding text overlay at frame {i}: {str(e)}")
                    
                    # Write frame to video
                    out.write(frame)
                    
                    # Update progress
                    progress = (i + 1) / len(frames_data)
                    video_progress.progress(progress)
                    
                except Exception as e:
                    st.warning(f"Error processing frame {i}: {str(e)}")
                    continue
        
        except Exception as e:
            st.error(f"Critical error during video creation: {str(e)}")
            
        finally:
            out.release()
            video_progress.empty()
        
        # Verify the output file exists and has content
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            st.success(f"Video created successfully: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
            
            # Check if we need to improve browser compatibility
            needs_conversion = used_codec not in ['H264', 'avc1']
            
            if needs_conversion:
                try:
                    import subprocess
                    ffmpeg_output = output_path.replace('.mp4', '_browser_compatible.mp4')
                    
                    # FFmpeg command for maximum browser compatibility
                    cmd = [
                        'ffmpeg', '-y', '-i', output_path,
                        '-c:v', 'libx264',
                        '-profile:v', 'baseline',
                        '-level', '3.0',
                        '-pix_fmt', 'yuv420p',
                        '-crf', '23',
                        '-preset', 'fast',
                        '-movflags', '+faststart',
                        '-f', 'mp4',
                        '-loglevel', 'error',
                        ffmpeg_output
                    ]
                    
                    st.info(f"Converting {used_codec} to browser-compatible H.264...")
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                    
                    if result.returncode == 0 and os.path.exists(ffmpeg_output) and os.path.getsize(ffmpeg_output) > 0:
                        st.success("Video converted for optimal browser compatibility!")
                        # Remove original file to save space
                        try:
                            os.remove(output_path)
                        except:
                            pass
                        return ffmpeg_output
                    else:
                        st.warning(f"FFmpeg conversion failed: {result.stderr[:200]}...")
                        st.warning("Using original video - it may not play in all browsers")
                        return output_path
                        
                except (ImportError, subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError) as e:
                    st.warning(f"FFmpeg not available ({str(e)[:100]}...)")
                    st.info("Original video may have compatibility issues in some browsers")
                    return output_path
            else:
                # Original codec should be browser-compatible
                st.info(f"Video created with {used_codec} codec - should be browser compatible")
                return output_path
        else:
            st.error("Failed to create video file or file is empty")
            return None

def create_web_compatible_video(input_path):
    """Create a web-compatible version of the video using FFmpeg"""
    try:
        import subprocess
        
        # Create output path
        output_path = input_path.replace('.mp4', '_web_compatible.mp4')
        
        # FFmpeg command for maximum browser compatibility
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-c:v', 'libx264',
            '-profile:v', 'baseline',
            '-level', '3.0',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            '-preset', 'medium',
            '-movflags', '+faststart',
            '-f', 'mp4',
            '-loglevel', 'error',
            output_path
        ]
        
        st.info("Converting video for web compatibility...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        
        if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return output_path
        else:
            st.error(f"FFmpeg conversion failed: {result.stderr}")
            return None
            
    except Exception as e:
        st.error(f"Error in web conversion: {str(e)}")
        return None

def create_sample_videos():
    """Create sample video directory and information"""
    sample_dir = Path("sample_videos")
    sample_dir.mkdir(exist_ok=True)
    
    # Information about sample videos (users would need to add actual videos)
    sample_info = [
        {"name": "person_walking.mp4", "description": "Person walking in a park"},
        {"name": "car_traffic.mp4", "description": "Cars in traffic"},
        {"name": "dog_playing.mp4", "description": "Dog playing in the yard"}
    ]
    
    return sample_info

def visualize_tracking_results(frames_data):
    """Visualize tracking results using Plotly"""
    if not frames_data:
        st.warning("No tracking data to visualize")
        return
    
    # Create detection count over time plot
    frame_indices = [data['frame_idx'] for data in frames_data]
    detection_counts = [data['detections_count'] for data in frames_data]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frame_indices,
        y=detection_counts,
        mode='lines+markers',
        name='Object Detections',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="Object Detection Count Over Time",
        xaxis_title="Frame Index",
        yaxis_title="Number of Detections",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create confidence distribution plot
    all_confidences = []
    for data in frames_data:
        for detection in data['detections']:
            all_confidences.append(detection['confidence'])
    
    if all_confidences:
        fig_hist = px.histogram(
            x=all_confidences,
            title="Detection Confidence Distribution",
            labels={'x': 'Confidence Score', 'y': 'Count'}
        )
        st.plotly_chart(fig_hist, use_container_width=True)

def main():
    st.title("🎯 Grounding SAM 2 Video Object Tracking")
    st.markdown("### Segment and track objects in videos using text prompts")
    
    # Initialize pipeline
    if 'pipeline' not in st.session_state:
        with st.spinner("Loading models..."):
            st.session_state.pipeline = GroundingSAM2Pipeline()
    
    # Sidebar
    st.sidebar.header("Configuration")
    if st.sidebar.button("🔄 Reload models"):
        with st.spinner("Reloading models..."):
            st.session_state.pipeline = GroundingSAM2Pipeline()
    
    # Text input for object description
    text_prompt = st.sidebar.text_input(
        "Object to track (e.g., 'person', 'car', 'dog')",
        value="person",
        help="Describe the object you want to track in the video"
    )
    
    st.sidebar.header("Visualization Options")
    
    # Visualization settings
    show_masks = st.sidebar.checkbox("Show segmentation masks", value=True)
    show_boxes = st.sidebar.checkbox("Show bounding boxes", value=True)
    show_labels = st.sidebar.checkbox("Show confidence labels", value=True)
    mask_opacity = st.sidebar.slider("Mask opacity", 0.1, 1.0, 0.5, 0.1)
    
    # Store visualization settings in session state
    st.session_state.viz_settings = {
        'show_masks': show_masks,
        'show_boxes': show_boxes,
        'show_labels': show_labels,
        'mask_opacity': mask_opacity
    }
    
    # Model status
    st.sidebar.header("🤖 Model Status")
    pipeline = st.session_state.pipeline
    
    if hasattr(pipeline, 'detection_method'):
        if pipeline.detection_method == "grounding_dino":
            st.sidebar.success("🎯 Active: Grounding DINO")
            if pipeline.sam_predictor is not None:
                st.sidebar.success("✨ Active: SAM Segmentation")
            else:
                st.sidebar.warning("⚠️ SAM: Using simple masks")
        elif pipeline.detection_method == "yolo":
            st.sidebar.warning("🔄 Fallback: YOLO Detection")
            st.sidebar.info("💡 Consider installing Grounding DINO for better text-based detection")
        else:
            st.sidebar.error("❌ No detection method available")

    # SAM2/SAM v1 availability details
    with st.sidebar.expander("Model details"):
        st.write(f"SAM2 available: {SAM2_AVAILABLE}")
        if not SAM2_AVAILABLE and SAM2_IMPORT_ERROR:
            st.caption(f"SAM2 import error: {SAM2_IMPORT_ERROR}")
        st.write(f"SAM v1 available: {SAM_AVAILABLE}")
        st.write(f"SAM2 predictor loaded: {pipeline.sam2_predictor is not None}")
        st.write(f"SAM v1 predictor loaded: {pipeline.sam_predictor is not None}")
    
    # Detection settings
    st.sidebar.header("🎯 Detection Settings")
    
    # For Grounding DINO
    if hasattr(pipeline, 'detection_method') and pipeline.detection_method == "grounding_dino":
        box_threshold = st.sidebar.slider("Box threshold", 0.1, 0.9, 0.25, 0.05)
        text_threshold = st.sidebar.slider("Text threshold", 0.1, 0.9, 0.25, 0.05)
        st.session_state.box_threshold = box_threshold
        st.session_state.text_threshold = text_threshold
    
    # Debug controls
    st.sidebar.header("🔧 Debug Options")
    debug_mode = st.sidebar.checkbox("Enable debug mode", value=True)
    st.session_state.debug_mode = debug_mode
    
    if debug_mode:
        show_all_detections = st.sidebar.checkbox("Show all detected objects (not just matching prompt)", value=True)
        st.session_state.show_all_detections = show_all_detections
        
        confidence_threshold = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.25, 0.05)
        st.session_state.confidence_threshold = confidence_threshold
        
        st.sidebar.info("💡 Debug mode is enabled to help diagnose detection issues. Disable after troubleshooting.")
    
    # Debug info in sidebar
    with st.sidebar.expander("🔧 Debug Info"):
        st.write("Current settings:")
        st.json(st.session_state.viz_settings)
        
        if hasattr(st.session_state, 'tracking_results') and st.session_state.tracking_results:
            total_detections = sum(r['detections_count'] for r in st.session_state.tracking_results)
            st.write(f"total detections across all frames: {total_detections}")
            
            # Show first few detections for debugging
            first_detection_frame = next((r for r in st.session_state.tracking_results if r['detections_count'] > 0), None)
            if first_detection_frame:
                st.write("First detection example:")
                st.json(first_detection_frame['detections'][0])
            
            # Show all detected classes
            if debug_mode:
                all_classes = set()
                for frame_data in st.session_state.tracking_results:
                    for detection in frame_data.get('all_detections', []):
                        all_classes.add(detection['class'])
                
                if all_classes:
                    st.write("All detected classes in video:")
                    for cls in sorted(all_classes):
                        st.write(f"- {cls}")
                else:
                    st.write("No objects detected in any frame")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Video Input")
        
        # Video upload
        uploaded_file = st.file_uploader(
            "Upload a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to process"
        )
        
        # Sample videos section
        st.subheader("Sample Videos")
        sample_videos = create_sample_videos()
        
        for video_info in sample_videos:
            st.info(f"📁 {video_info['name']}: {video_info['description']}")
        
        st.markdown("*Note: Add sample videos to the `sample_videos/` directory*")
    
    with col2:
        st.header("Processing Results")
        
        if uploaded_file is not None:
            # Save uploaded file temporarily with safe filename
            try:
                # Create a safe temporary file with proper extension
                file_extension = os.path.splitext(uploaded_file.name)[1] or '.mp4'
                
                # Create temporary directory if it doesn't exist
                temp_dir = Path("temp_videos")
                temp_dir.mkdir(exist_ok=True)
                
                # Generate safe filename
                import hashlib
                import time
                safe_name = hashlib.md5(f"{uploaded_file.name}_{time.time()}".encode()).hexdigest()
                temp_video_path = temp_dir / f"{safe_name}{file_extension}"
                
                # Write file safely
                with open(temp_video_path, 'wb') as tmp_file:
                    uploaded_file.seek(0)  # Reset file pointer
                    tmp_file.write(uploaded_file.read())
                
                # Verify file was written correctly
                if not temp_video_path.exists() or temp_video_path.stat().st_size == 0:
                    st.error("Failed to save uploaded video file")
                    temp_video_path = None
                else:
                    st.success(f"Video uploaded: {temp_video_path.name} ({temp_video_path.stat().st_size / (1024*1024):.2f} MB)")
                
            except Exception as e:
                st.error(f"Error saving uploaded file: {str(e)}")
                temp_video_path = None
            
            if temp_video_path and temp_video_path.exists():
                try:
                    # Display video
                    st.video(uploaded_file)
                    
                    # Process button
                    if st.button("🚀 Process Video", type="primary"):
                        with st.spinner(f"Processing video to detect '{text_prompt}'..."):
                            # Show current debug settings
                            if st.session_state.get('debug_mode', False):
                                st.info(f"🔧 Debug mode: Show all detections = {st.session_state.get('show_all_detections', False)}, Confidence threshold = {st.session_state.get('confidence_threshold', 0.25)}")
                            
                            # Process video
                            results = st.session_state.pipeline.process_video(
                                str(temp_video_path), text_prompt
                            )
                            
                            # Store results in session state
                            st.session_state.tracking_results = results
                            
                            if results:
                                frames_with_detections = len([r for r in results if r['detections_count'] > 0])
                                total_detections = sum(r['detections_count'] for r in results)
                                
                                # Show detailed results
                                if frames_with_detections > 0:
                                    st.success(f"✅ Processing complete! Found {total_detections} total detections in {frames_with_detections} frames out of {len(results)} total frames.")
                                else:
                                    st.warning(f"⚠️ Processing complete but no matching detections found in {len(results)} frames.")
                                    
                                    # Show what was actually detected for debugging
                                    if st.session_state.get('debug_mode', False):
                                        all_detected_classes = set()
                                        for frame_data in results:
                                            for detection in frame_data.get('all_detections', []):
                                                all_detected_classes.add(detection['class'])
                                        
                                        if all_detected_classes:
                                            st.info(f"🔍 Detected classes in video: {', '.join(sorted(all_detected_classes))}")
                                            st.info(f"💡 Try using one of these class names as your search prompt, or enable 'Show all detected objects' in debug options.")
                                        else:
                                            st.warning("🔍 No objects detected by YOLO model in any frame.")
                            else:
                                st.warning("No frames were processed successfully.")
                    
                    # Generate annotated video button
                    if hasattr(st.session_state, 'tracking_results') and st.session_state.tracking_results:
                        if st.button("🎨 Generate Annotated Video", type="secondary"):
                            with st.spinner("Creating annotated video with segmentation masks..."):
                                # Create output directory
                                output_dir = Path("output_videos")
                                output_dir.mkdir(exist_ok=True)
                                
                                # Generate unique filename
                                import time
                                timestamp = int(time.time())
                                output_path = output_dir / f"annotated_video_{timestamp}.mp4"
                                
                                # Create annotated video with visualization settings
                                viz_settings = getattr(st.session_state, 'viz_settings', {})
                                annotated_video_path = st.session_state.pipeline.create_annotated_video(
                                    st.session_state.tracking_results, str(output_path), 
                                    fps=30, viz_settings=viz_settings
                                )
                                
                                if annotated_video_path and os.path.exists(annotated_video_path):
                                    st.session_state.annotated_video_path = annotated_video_path
                                    st.success("Annotated video created successfully!")
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
            else:
                st.error("Could not save uploaded video file. Please try again.")
        
        # Display annotated video if available (outside the uploaded file check)
        if hasattr(st.session_state, 'annotated_video_path') and os.path.exists(st.session_state.annotated_video_path):
            st.subheader("🎬 Annotated Video with Segmentation")
            
            try:
                # Check file size before reading
                file_size = os.path.getsize(st.session_state.annotated_video_path)
                if file_size > 0:
                    st.caption(f"Video file: {os.path.basename(st.session_state.annotated_video_path)} ({file_size / (1024*1024):.2f} MB)")
                    
                    # Try multiple display methods
                    video_displayed = False
                    
                    # Method 1: Direct file path (works for local files)
                    try:
                        st.video(st.session_state.annotated_video_path)
                        video_displayed = True
                        st.success("✅ Video loaded successfully!")
                    except Exception as e1:
                        st.warning(f"Method 1 failed: {str(e1)}")
                        
                        # Method 2: Read as bytes
                        try:
                            with open(st.session_state.annotated_video_path, 'rb') as video_file:
                                video_bytes = video_file.read()
                            
                            if len(video_bytes) > 0:
                                st.video(video_bytes)
                                video_displayed = True
                                st.success("✅ Video loaded as bytes!")
                            else:
                                st.error("Video file is empty")
                                
                        except Exception as e2:
                            st.error(f"Method 2 failed: {str(e2)}")
                            
                            # Method 3: Create web-compatible version
                            try:
                                web_compatible_path = create_web_compatible_video(st.session_state.annotated_video_path)
                                if web_compatible_path and os.path.exists(web_compatible_path):
                                    with open(web_compatible_path, 'rb') as video_file:
                                        video_bytes = video_file.read()
                                    st.video(video_bytes)
                                    video_displayed = True
                                    st.success("✅ Video converted for web compatibility!")
                                else:
                                    st.error("Failed to create web-compatible video")
                            except Exception as e3:
                                st.error(f"Method 3 failed: {str(e3)}")
                    
                    # Download button (always available)
                    try:
                        with open(st.session_state.annotated_video_path, 'rb') as video_file:
                            video_bytes = video_file.read()
                        
                        st.download_button(
                            label="📥 Download Annotated Video",
                            data=video_bytes,
                            file_name=f"segmented_video_{text_prompt.replace(' ', '_')}.mp4",
                            mime="video/mp4"
                        )
                    except Exception as e:
                        st.error(f"Failed to prepare download: {str(e)}")
                    
                    if video_displayed:
                        st.info("💡 The annotated video shows detected objects with colored segmentation masks and bounding boxes.")
                    else:
                        st.error("❌ Could not display video in browser. Please use the download button.")
                        
                        # Show video info as fallback
                        st.write("**Video Information:**")
                        try:
                            import subprocess
                            result = subprocess.run(['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', st.session_state.annotated_video_path], 
                                                  capture_output=True, text=True, timeout=10)
                            if result.returncode == 0:
                                import json
                                info = json.loads(result.stdout)
                                if 'streams' in info and len(info['streams']) > 0:
                                    video_stream = info['streams'][0]
                                    st.write(f"- Format: {info.get('format', {}).get('format_name', 'Unknown')}")
                                    st.write(f"- Duration: {float(info.get('format', {}).get('duration', 0)):.2f} seconds")
                                    st.write(f"- Resolution: {video_stream.get('width', 'Unknown')}x{video_stream.get('height', 'Unknown')}")
                                    st.write(f"- Codec: {video_stream.get('codec_name', 'Unknown')}")
                        except:
                            st.write("Could not retrieve video information")
                else:
                    st.error("Generated video file is empty. Please try processing again.")
            
            except Exception as e:
                st.error(f"Error loading video: {str(e)}")
                st.info("You can try downloading the video file directly or regenerating it.")
                
                # Fallback: Show sample frames
                if hasattr(st.session_state, 'tracking_results') and st.session_state.tracking_results:
                    st.subheader("📸 Sample Frames with Annotations")
                    
                    # Show every 10th frame as images
                    sample_frames = [r for i, r in enumerate(st.session_state.tracking_results) if i % 10 == 0 and r['detections_count'] > 0]
                    
                    if sample_frames:
                        cols = st.columns(min(3, len(sample_frames)))
                        
                        # Debug: Show detection details
                        with st.expander("🔍 Detection Debug Info"):
                            for frame_data in sample_frames[:3]:
                                st.write(f"**Frame {frame_data['frame_idx']}:**")
                                for i, detection in enumerate(frame_data['detections']):
                                    st.write(f"  Detection {i+1}: {detection}")
                        
                        for i, frame_data in enumerate(sample_frames[:6]):  # Show max 6 frames
                            with cols[i % 3]:
                                # Create annotated frame
                                frame = frame_data['frame'].copy()
                                viz_settings = getattr(st.session_state, 'viz_settings', {
                                    'show_masks': True,
                                    'show_boxes': True,
                                    'show_labels': True,
                                    'mask_opacity': 0.5
                                })
                                
                                # Draw detections on frame with different colors
                                colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
                                for j, detection in enumerate(frame_data['detections']):
                                    color = colors[j % len(colors)]
                                    frame = st.session_state.pipeline.draw_detection_on_frame(frame, detection, color, viz_settings)
                                
                                # Convert BGR to RGB for display
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                
                                # Show detection details in caption
                                detection_info = ", ".join([f"{d['class']}({d['confidence']:.2f})" for d in frame_data['detections']])
                                caption = f"Frame {frame_data['frame_idx']}: {detection_info}"
                                
                                st.image(frame_rgb, caption=caption, use_container_width=True)
                    else:
                        st.info("No frames with detections found for preview.")
                
        # Display results if available (outside the annotated video section)
        if hasattr(st.session_state, 'tracking_results'):
            st.subheader("📊 Tracking Analysis")
            visualize_tracking_results(st.session_state.tracking_results)
            
            # Display summary statistics
            total_frames = len(st.session_state.tracking_results)
            frames_with_detections = len([r for r in st.session_state.tracking_results if r['detections_count'] > 0])
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Frames", total_frames)
            with col_b:
                st.metric("Frames with Detections", frames_with_detections)
            with col_c:
                detection_rate = (frames_with_detections / total_frames * 100) if total_frames > 0 else 0
                st.metric("Detection Rate", f"{detection_rate:.1f}%")
            
            # Additional visualization: Detection heatmap
            if frames_with_detections > 0:
                with st.expander("🔥 Detection Heatmap"):
                    # Create a simple heatmap showing detection density
                    detection_frames = [r['frame_idx'] for r in st.session_state.tracking_results if r['detections_count'] > 0]
                    
                    fig_heatmap = go.Figure(data=go.Histogram(
                        x=detection_frames,
                        nbinsx=20,
                        name='Detection Density'
                    ))
                    
                    fig_heatmap.update_layout(
                        title="Detection Density Across Video Timeline",
                        xaxis_title="Frame Index",
                        yaxis_title="Detection Frequency",
                        bargap=0.1
                    )
                    
                    st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Clean up temporary file when done
        if 'temp_video_path' in locals() and temp_video_path and temp_video_path.exists():
            try:
                temp_video_path.unlink()
            except Exception as e:
                st.warning(f"Could not clean up temporary file: {str(e)}")
    
    # Pipeline visualization section
    st.header("🔄 Processing Pipeline")
    
    with st.expander("View Pipeline Details"):
        st.markdown("""
        ### Grounding SAM 2 Pipeline
        
        1. **Video Input**: Upload or select a sample video
        2. **Text Prompt**: Specify the object to track using natural language
        3. **Object Detection**: Use GroundingDINO for text-conditioned object detection
        4. **Segmentation**: Apply SAM 2 for precise object segmentation
        5. **Tracking**: Track detected objects across video frames
        6. **Visualization**: Display results using interactive Plotly charts
        """)
        
        # Pipeline flow diagram
        pipeline_steps = ["Video Input", "Text Prompt", "Object Detection", "Segmentation", "Tracking", "Visualization"]
        
        fig_pipeline = go.Figure()
        
        for i, step in enumerate(pipeline_steps):
            fig_pipeline.add_trace(go.Scatter(
                x=[i],
                y=[0],
                mode='markers+text',
                marker=dict(size=30, color='lightblue'),
                text=step,
                textposition="middle center",
                showlegend=False
            ))
            
            if i < len(pipeline_steps) - 1:
                fig_pipeline.add_annotation(
                    x=i+0.4, y=0,
                    ax=i+0.6, ay=0,
                    xref='x', yref='y',
                    axref='x', ayref='y',
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor='gray'
                )
        
        fig_pipeline.update_layout(
            title="Processing Pipeline Flow",
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            height=200,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig_pipeline, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit • Powered by Grounding SAM 2")

if __name__ == "__main__":
    main()
