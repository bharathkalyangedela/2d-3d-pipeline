# 2D to 3D Video Conversion Pipeline

This pipeline converts 2D videos to 3D using depth maps and ProPainter for inpainting.

## Prerequisites

- Python 3.8+
- OpenCV
- NumPy
- OpenEXR (for .exr depth files)
- tqdm
- CUDA-compatible GPU (recommended)

## Installation

1. Clone this repository with submodules:
```bash
git clone --recursive <your-repo-url>
cd <your-repo-name>
```

2. Install dependencies:
```bash
pip install opencv-python numpy openexr tqdm
```

3. Set up ProPainter:
```bash
cd ProPainter
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python integrated_3d_conversion.py \
    --video <input_video_path> \
    --depth_folder <depth_maps_path> \
    --output_dir <output_directory>
```

### Parameters

- `--video`: Path to input 2D video
- `--depth_folder`: Path to depth maps folder or single .npy/.exr file
- `--output_dir`: Base output directory
- `--shift_amount`: 3D effect strength (10-100, default: 50)
- `--batch_size`: Number of frames per batch (default: 60)
- `--mask_dilation_kernel`: Kernel size for mask dilation (1-5, default: 1)
- `--mask_dilation_iterations`: Number of dilation iterations (1-5, default: 1)
- `--neighbor_length`: Number of neighbor frames to consider (default: 10)
- `--ref_stride`: Stride for reference frames (default: 10)
- `--raft_iter`: Number of RAFT iterations (default: 20)
- `--save_fps`: Output video FPS (default: 24)

### Output

The pipeline creates the following directory structure:
```
output_dir/
├── stereo_output/     # Stereo image pairs
├── masks_left/        # Left view masks
├── masks_right/       # Right view masks
├── results_left/      # Left view results
├── results_right/     # Right view results
├── combined_left/     # Combined left view
├── combined_right/    # Combined right view
├── anaglyph_output.mp4    # Red-cyan anaglyph video
└── stereo_pair_output.mp4 # Side-by-side stereo video
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- ProPainter: https://github.com/sczhou/ProPainter 