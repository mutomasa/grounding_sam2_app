# Grounding SAM 2 Video Object Tracking Application

🎯 A Streamlit-based application for object detection, segmentation, and tracking in videos using Grounding SAM 2 technology.

## Overview

This application combines the power of Grounding SAM 2 (Segment Anything Model 2) with natural language object detection to provide an intuitive interface for video analysis. Users can upload videos and specify objects to track using simple text descriptions.

## Features

- 📹 **Video Upload**: Support for multiple video formats (MP4, AVI, MOV, MKV)
- 🔍 **Text-based Object Detection**: Describe objects in natural language
- 🎯 **Precise Segmentation**: Leverage SAM 2 for accurate object boundaries
- 📊 **Real-time Tracking**: Track objects across video frames
- 📈 **Interactive Visualization**: Plotly-powered charts and analytics
- 🖥️ **User-friendly Interface**: Clean Streamlit web interface

## Technical Architecture

### Grounding SAM 2 Technology

**Grounding SAM 2** represents a significant advancement in vision-language understanding, combining:

#### Model Architecture

1. **GroundingDINO**: 
   - Text-conditioned object detection model
   - Transforms natural language queries into visual object detection
   - Uses transformer architecture with cross-modal attention mechanisms
   - Enables zero-shot detection of objects described in text

2. **Segment Anything Model 2 (SAM 2)**:
   - Advanced segmentation model by Meta AI
   - Provides pixel-perfect object boundaries
   - Supports various prompt types (points, boxes, masks)
   - Optimized for video temporal consistency

3. **Video Tracking Pipeline**:
   - Combines detection and segmentation for robust tracking
   - Maintains object identity across frames
   - Handles occlusions and re-identification

#### Advanced Features

- **Cross-modal Understanding**: Bridges natural language and computer vision
- **Zero-shot Capabilities**: Detects objects without specific training
- **Temporal Consistency**: Maintains tracking across video sequences
- **High-quality Segmentation**: Pixel-level accuracy for object boundaries
- **Real-time Processing**: Optimized for efficient video analysis

### Technical Implementation

```
Text Prompt → GroundingDINO → Object Detection → SAM 2 → Segmentation → Tracking
```

#### Key Components:

1. **Text Encoder**: Processes natural language descriptions
2. **Vision Encoder**: Extracts visual features from video frames
3. **Cross-modal Fusion**: Aligns text and visual representations
4. **Object Detector**: Localizes objects based on text queries
5. **Segmentation Model**: Generates precise object masks
6. **Tracker**: Maintains object identity across frames

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- UV package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd grounding_sam2_app
```

2. Install dependencies using UV:
```bash
uv init
uv add streamlit plotly opencv-python pillow torch torchvision numpy pandas matplotlib seaborn transformers supervision ultralytics segment-anything
```

3. Run the application:
```bash
uv run streamlit run main.py
```

## Usage

### Basic Workflow

1. **Start the Application**:
   ```bash
   uv run streamlit run main.py
   ```

2. **Upload a Video**:
   - Click "Browse files" to upload a video
   - Supported formats: MP4, AVI, MOV, MKV

3. **Specify Object to Track**:
   - Enter a text description in the sidebar
   - Examples: "person", "car", "dog", "red car", "walking person"

4. **Process Video**:
   - Click "🚀 Process Video" button
   - Wait for processing to complete

5. **View Results**:
   - Interactive charts showing detection statistics
   - Frame-by-frame tracking results
   - Processing pipeline visualization

### Advanced Features

- **Pipeline Visualization**: View the complete processing workflow
- **Detection Analytics**: Confidence distributions and temporal analysis
- **Sample Videos**: Pre-configured test cases for demonstration

## Model Details

### GroundingDINO

- **Architecture**: Transformer-based detection model
- **Training**: Large-scale vision-language datasets
- **Capabilities**: Open-vocabulary object detection
- **Performance**: High accuracy on diverse object categories

### SAM 2

- **Model Size**: Multiple variants (Base, Large, Huge)
- **Training Data**: SA-1B dataset (1+ billion masks)
- **Segmentation Quality**: State-of-the-art mask precision
- **Video Support**: Temporal consistency across frames

## Performance Considerations

### Hardware Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **CPU**: Multi-core processor for video processing
- **RAM**: 16GB+ for large video files
- **Storage**: SSD recommended for video I/O

### Optimization Tips

- Use lower resolution videos for faster processing
- Limit video length for real-time analysis
- Adjust detection confidence thresholds
- Enable GPU acceleration when available

## Use Cases

### Professional Applications

- **Security & Surveillance**: Track specific individuals or objects
- **Sports Analysis**: Analyze player movements and strategies
- **Wildlife Monitoring**: Track animals in natural habitats
- **Industrial Inspection**: Monitor equipment and processes

### Research Applications

- **Computer Vision Research**: Evaluate tracking algorithms
- **Behavioral Studies**: Analyze object interactions
- **Dataset Creation**: Generate annotated video datasets
- **Benchmark Testing**: Compare detection models

## Technical Specifications

### Supported Video Codecs

- H.264/AVC
- H.265/HEVC
- VP9
- AV1 (experimental)

### Processing Capabilities

- **Frame Rate**: Up to 30 FPS
- **Resolution**: Up to 4K (4096×2160)
- **Duration**: No theoretical limit (memory dependent)
- **Batch Processing**: Multiple videos in sequence

## API Reference

### Core Classes

#### `GroundingSAM2Pipeline`

Main processing pipeline for video analysis.

**Methods:**
- `load_models()`: Initialize detection and segmentation models
- `detect_objects_in_frame(frame, text_prompt)`: Detect objects in single frame
- `process_video(video_path, text_prompt)`: Process entire video

### Visualization Functions

#### `visualize_tracking_results(frames_data)`

Generate interactive charts for tracking analysis.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Meta AI**: For the Segment Anything Model 2
- **IDEA Research**: For GroundingDINO implementation
- **Ultralytics**: For YOLO integration
- **Streamlit**: For the web application framework

## References

- [Grounding SAM 2 Repository](https://github.com/IDEA-Research/Grounded-SAM-2)
- [SAM 2 Paper](https://arxiv.org/abs/2401.12741)
- [GroundingDINO Paper](https://arxiv.org/abs/2303.05499)
- [Segment Anything Project](https://segment-anything.com/)


