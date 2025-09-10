# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Environment Setup
```bash
# Install dependencies using uv
uv install

# Run the Streamlit application  
uv run streamlit run main.py
```

### Video Processing
- The app processes videos up to 100 frames by default to prevent memory issues
- Supported video formats: MP4, AVI, MOV, MKV
- Output videos are saved to `output_videos/` directory
- Temporary videos are stored in `temp_videos/` directory

### Model Management
- SAM checkpoints are automatically downloaded to `checkpoints/` directory
- The app attempts to load models in this priority order:
  1. Grounding DINO + SAM2 (preferred)
  2. Grounding DINO + SAM v1 (fallback)
  3. YOLO (final fallback)

## Code Architecture

### Core Pipeline (`GroundingSAM2Pipeline` class)
The application follows a multi-stage processing pipeline:

1. **Model Loading** (`load_models()`): Attempts to load GroundingDINO, SAM2/SAM1, and YOLO models
2. **Object Detection** (`detect_objects_in_frame()`): Uses text prompts to detect objects via GroundingDINO or YOLO fallback
3. **Segmentation** (`generate_segmentation_mask()`): Creates precise object masks using SAM2/SAM1
4. **Video Processing** (`process_video()`): Processes video frame-by-frame with detection and tracking
5. **Visualization** (`create_annotated_video()`): Generates output video with segmentation overlays

### Model Integration Strategy
- **Primary**: Grounding DINO for text-conditioned detection + SAM2 for segmentation
- **Fallback 1**: Grounding DINO + SAM v1
- **Fallback 2**: YOLO detection with simple elliptical masks
- Models are loaded dynamically based on availability

### Error Handling and Robustness
- Multiple video codec fallback options (H264 → avc1 → mp4v → XVID → MJPG)
- FFmpeg post-processing for browser compatibility
- Graceful degradation when models fail to load
- Progress tracking for long-running operations

### Session State Management
Key session variables:
- `pipeline`: Main processing pipeline instance
- `tracking_results`: Frame-by-frame detection results
- `annotated_video_path`: Path to generated output video
- `viz_settings`: Visualization options (masks, boxes, labels, opacity)

### File Organization
- `main.py`: Single-file Streamlit application with all functionality
- `checkpoints/`: Model files (SAM weights)
- `sample_videos/`: Example input videos
- `temp_videos/`: Temporary uploaded files
- `output_videos/`: Generated annotated videos

## Development Notes

### Text Prompts
- Grounding DINO requires prompts to end with a period
- Examples: "person.", "red car.", "walking dog."
- YOLO fallback uses string matching against COCO class names

### Video Processing Limitations
- Maximum 100 frames processed to prevent memory issues
- Frame dimensions adjusted to even numbers for codec compatibility
- Consecutive frame read failures cause early termination (max 10 failures)

### Dependencies
- Core ML: `torch`, `torchvision`, `transformers`, `ultralytics`
- Computer Vision: `opencv-python`, `segment-anything`, `sam-2`, `supervision`  
- UI/Visualization: `streamlit`, `plotly`, `matplotlib`, `seaborn`
- Utilities: `numpy`, `pandas`, `pillow`

### Debug Features
- Debug mode shows all detected objects and confidence scores
- Sidebar displays model loading status and detection statistics
- Frame-by-frame detection logging available in debug mode