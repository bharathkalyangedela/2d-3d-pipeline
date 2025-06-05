import os
import cv2
import numpy as np
import subprocess
import shutil
from tqdm import tqdm
import argparse
import logging
from datetime import datetime
import time

def setup_logging(output_dir):
    """Setup logging configuration"""
    log_file = os.path.join(output_dir, f'conversion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_directories(base_dir):
    """Create necessary directories for the pipeline"""
    dirs = {
        'stereo_output': os.path.join(base_dir, 'stereo_output'),
        'masks_left': os.path.join(base_dir, 'masks_left'),
        'masks_right': os.path.join(base_dir, 'masks_right'),
        'results_left': os.path.join(base_dir, 'results_left'),
        'results_right': os.path.join(base_dir, 'results_right'),
        'combined_left': os.path.join(base_dir, 'combined_left'),
        'combined_right': os.path.join(base_dir, 'combined_right')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")
    
    return dirs

def validate_inputs(video_path, depth_path):
    """Validate input files and formats"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth file/folder not found: {depth_path}")
    
    # Check video format
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Invalid video file: {video_path}")
    cap.release()
    
    # Check if depth_path is a file or directory
    if os.path.isfile(depth_path):
        if not depth_path.endswith(('.npy', '.exr')):
            raise ValueError(f"Unsupported depth file format: {depth_path}")
        print(f"Detected single depth file: {depth_path}")
        return True, depth_path
    else:
        # It's a directory, check for depth files
        depth_files = [f for f in os.listdir(depth_path) if f.endswith(('.npy', '.exr'))]
        if not depth_files:
            raise ValueError(f"No depth maps found in: {depth_path}")
        
        # Check if it's a single large .npy file
        single_npy = any(f.endswith('.npy') and os.path.getsize(os.path.join(depth_path, f)) > 1024*1024 for f in depth_files)
        if single_npy:
            print("Detected single large .npy file for depth maps")
            return True, os.path.join(depth_path, [f for f in depth_files if f.endswith('.npy')][0])
        else:
            print("Detected individual depth map files")
            return False, depth_path

def extract_frames(video_path, output_dir):
    """Extract frames from video"""
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_path = os.path.join(output_dir, f'frame_{frame_count:04d}.png')
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frame_count += 1
    
    cap.release()
    return frame_count

def read_exr(exr_path, channel='Z'):
    """Read depth data from .exr file"""
    import OpenEXR
    import Imath
    
    exr_file = OpenEXR.InputFile(exr_path)
    header = exr_file.header()
    dw = header['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    
    depth_str = exr_file.channel(channel)
    depth = np.frombuffer(depth_str, dtype=np.float32)
    depth = depth.reshape(size[1], size[0])
    return depth

def shift_image_with_transparency(img, depth_img, shift_amount, direction='left'):
    """Create stereo pair with transparency based on depth"""
    height, width = img.shape[:2]
    result = np.zeros((height, width, 4), dtype=np.uint8)
    
    depth_norm = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
    shifts = (depth_norm * shift_amount).astype(np.int32)
    
    if direction == 'left':
        for y in range(height):
            for x in range(width):
                shift = shifts[y, x]
                if x + shift < width:
                    result[y, x + shift] = np.append(img[y, x], 255)
    else:  # right
        for y in range(height):
            for x in range(width):
                shift = shifts[y, x]
                if x - shift >= 0:
                    result[y, x - shift] = np.append(img[y, x], 255)
    
    return result

def create_masks_from_transparent(input_dir, output_mask_dir, dilation_kernel_size=1, dilation_iterations=1):
    """Create masks from transparent areas with adjustable dilation"""
    os.makedirs(output_mask_dir, exist_ok=True)
    
    # Determine if we're processing left or right view based on output directory name
    is_left_view = 'left' in output_mask_dir.lower()
    view_prefix = 'left_' if is_left_view else 'right_'
    
    # Sort files to ensure correct order
    image_files = sorted([f for f in os.listdir(input_dir) if f.startswith(view_prefix) and f.endswith('.png')])
    
    for frame_count, img_file in enumerate(image_files):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        
        if img.shape[2] == 4:
            alpha = img[:, :, 3]
            mask = np.zeros_like(alpha)
            mask[alpha == 0] = 255
            
            kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=dilation_iterations)
            
            # Use frame_count for mask numbering
            mask_path = os.path.join(output_mask_dir, f'mask_{frame_count:04d}.png')
            cv2.imwrite(mask_path, mask)
            
            if frame_count % 10 == 0:
                print(f"Created mask for frame {frame_count} ({view_prefix[:-1]} view)")

def check_batch_outputs(batch_num, output_base, batch_size):
    """Check if ProPainter outputs already exist for a batch"""
    batch_dir = os.path.join(output_base, f'batch_{batch_num}')
    batch_results_dir = os.path.join(batch_dir, f'batch_{batch_num}')
    
    if not os.path.exists(output_base):
        print(f"Output base directory does not exist: {output_base}")
        return False
        
    if os.path.exists(batch_results_dir):
        # Look for inpainted video file
        video_files = [f for f in os.listdir(batch_results_dir) if f == 'inpaint_out.mp4']
        if video_files:
            print(f"Found ProPainter inpainted output for batch {batch_num}")
            return True
    return False

def extract_frames_from_video(video_path, output_dir, start_frame=0):
    """Extract frames from a video file"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    frame_count = start_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_path = os.path.join(output_dir, f'frame_{frame_count:04d}.png')
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    
    cap.release()
    return frame_count - start_frame

def process_batch(batch_num, input_base, mask_base, output_base, batch_size=60, propainter_params=None, view_type='left'):
    """Process a batch of frames using ProPainter"""
    # Ensure output directory exists
    os.makedirs(output_base, exist_ok=True)
    
    # Check if batch already processed
    if check_batch_outputs(batch_num, output_base, batch_size):
        print(f"Skipping batch {batch_num} - already processed")
        return
    
    # Get total number of frames
    frame_files = sorted([f for f in os.listdir(input_base) if f.startswith(f'{view_type}_') and f.endswith('.png')])
    total_frames = len(frame_files)
    
    # Calculate start and end indices for this batch
    start_idx = batch_num * batch_size
    end_idx = min(start_idx + batch_size, total_frames)
    
    # Skip if we've processed all frames
    if start_idx >= total_frames:
        print(f"Batch {batch_num} skipped - all frames processed")
        return
    
    # Calculate actual batch size for this batch
    actual_batch_size = end_idx - start_idx
    
    # If this is the last batch and it's too small (less than 2 frames), merge with previous batch
    if actual_batch_size < 2 and batch_num > 0:
        print(f"Last batch too small ({actual_batch_size} frames), merging with previous batch")
        return
    
    batch_dir = os.path.abspath(f"{input_base}/batches/batch_{batch_num}")
    output_dir = os.path.abspath(f"{output_base}/batch_{batch_num}")
    batch_mask_dir = os.path.abspath(f"{mask_base}/batches/batch_{batch_num}")
    
    # Clean up any existing incomplete batch directories
    for d in [batch_dir, batch_mask_dir, output_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
    
    os.makedirs(batch_dir, exist_ok=True)
    os.makedirs(batch_mask_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Created batch directories for {view_type} view:")
    print(f"  - Batch dir: {batch_dir}")
    print(f"  - Output dir: {output_dir}")
    print(f"  - Mask dir: {batch_mask_dir}")
    
    # Copy frames and masks to batch directories
    for i in range(actual_batch_size):
        idx = start_idx + i
        
        # Copy frame with consistent naming
        frame_src = os.path.join(input_base, f'{view_type}_{idx:04d}.png')
        frame_dst = os.path.join(batch_dir, f'frame_{i:04d}.png')
        if os.path.exists(frame_src):
            shutil.copy(frame_src, frame_dst)
        else:
            print(f"Warning: Frame file not found: {frame_src}")
            continue
        
        # Copy mask with consistent naming
        mask_src = os.path.join(mask_base, f'mask_{idx:04d}.png')
        mask_dst = os.path.join(batch_mask_dir, f'mask_{i:04d}.png')
        if os.path.exists(mask_src):
            shutil.copy(mask_src, mask_dst)
        else:
            print(f"Warning: Mask file not found: {mask_src}")
    
    try:
        # Run ProPainter with properly quoted paths
        cmd = f"cd ProPainter && python inference_propainter.py " \
              f"-i \"{batch_dir}\" " \
              f"-m \"{batch_mask_dir}\" " \
              f"-o \"{output_dir}\" " \
              f"--neighbor_length {propainter_params['neighbor_length']} " \
              f"--ref_stride {propainter_params['ref_stride']} " \
              f"--fp16 " \
              f"--mask_dilation 0 " \
              f"--save_fps {propainter_params['save_fps']} " \
              f"--raft_iter {propainter_params['raft_iter']}"
        
        print(f"Running ProPainter command for {view_type} view: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        
        # Wait a moment for files to be written
        time.sleep(2)
        
        # Verify the output was created successfully
        batch_results_dir = os.path.join(output_dir, f'batch_{batch_num}')
        if not os.path.exists(batch_results_dir):
            print(f"Warning: ProPainter output directory not found: {batch_results_dir}")
            raise RuntimeError(f"ProPainter did not create output directory for batch {batch_num}")
            
        # Look for inpainted video file
        if not os.path.exists(os.path.join(batch_results_dir, 'inpaint_out.mp4')):
            print(f"Warning: Inpainted video not found in {batch_results_dir}")
            raise RuntimeError(f"ProPainter did not generate inpainted video for batch {batch_num}")
            
        print(f"Successfully processed batch {batch_num} for {view_type} view")
            
    except Exception as e:
        print(f"Error processing batch {batch_num} for {view_type} view: {str(e)}")
        # Clean up failed batch
        for d in [batch_dir, batch_mask_dir, output_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)
        raise
    finally:
        # Clean up temporary directories
        for d in [batch_dir, batch_mask_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)

def combine_batches(batch_base, output_dir):
    """Combine processed batches into a single video"""
    batch_dirs = sorted([d for d in os.listdir(batch_base) if d.startswith('batch_')])
    
    if not batch_dirs:
        raise ValueError(f"No batch directories found in {batch_base}")
    
    print(f"Found {len(batch_dirs)} batch directories to combine")
    
    # Create temporary directory for extracted frames
    temp_dir = os.path.join(output_dir, 'temp_frames')
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        frame_count = 0
        for batch_dir in batch_dirs:
            batch_path = os.path.join(batch_base, batch_dir)
            batch_results_path = os.path.join(batch_path, batch_dir)
            
            if not os.path.exists(batch_results_path):
                print(f"Warning: Batch results directory not found: {batch_results_path}")
                continue
                
            # Look for inpainted video file
            video_path = os.path.join(batch_results_path, 'inpaint_out.mp4')
            if not os.path.exists(video_path):
                print(f"Warning: Inpainted video not found in {batch_results_path}")
                continue
                
            # Extract frames from video
            frames_extracted = extract_frames_from_video(video_path, temp_dir, frame_count)
            frame_count += frames_extracted
            print(f"Extracted {frames_extracted} frames from batch {batch_dir}")
        
        print(f"Combined {frame_count} frames from all batches")
        
        # Create video from combined frames
        if frame_count > 0:
            # Get first frame to determine dimensions
            first_frame = cv2.imread(os.path.join(temp_dir, 'frame_0000.png'))
            if first_frame is None:
                raise ValueError("Could not read first frame")
            
            height, width = first_frame.shape[:2]
            fps = 24
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_video = os.path.join(output_dir, 'combined_output.mp4')
            out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
            
            # Write all frames to video
            for i in tqdm(range(frame_count), desc="Creating combined video"):
                frame_path = os.path.join(temp_dir, f'frame_{i:04d}.png')
                frame = cv2.imread(frame_path)
                if frame is not None:
                    out.write(frame)
            
            out.release()
            print(f"Created combined video: {output_video}")
        else:
            raise ValueError("No frames were extracted from any batch")
            
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def create_anaglyph(left_dir, right_dir, output_path):
    """Create red-cyan anaglyph video from left and right views"""
    # Create temporary directories for frames
    temp_left = os.path.join(os.path.dirname(output_path), 'temp_left')
    temp_right = os.path.join(os.path.dirname(output_path), 'temp_right')
    temp_anaglyph = os.path.join(os.path.dirname(output_path), 'temp_anaglyph')
    os.makedirs(temp_left, exist_ok=True)
    os.makedirs(temp_right, exist_ok=True)
    os.makedirs(temp_anaglyph, exist_ok=True)
    
    try:
        # Extract frames from videos
        left_videos = sorted([f for f in os.listdir(left_dir) if f.endswith('.mp4')])
        right_videos = sorted([f for f in os.listdir(right_dir) if f.endswith('.mp4')])
        
        if not left_videos or not right_videos:
            raise ValueError(f"No video files found in {left_dir} or {right_dir}")
        
        # Extract frames from all videos
        for video in left_videos:
            video_path = os.path.join(left_dir, video)
            extract_frames_from_video(video_path, temp_left)
        
        for video in right_videos:
            video_path = os.path.join(right_dir, video)
            extract_frames_from_video(video_path, temp_right)
        
        # Get frame lists
        left_frames = sorted([f for f in os.listdir(temp_left) if f.endswith('.png')])
        right_frames = sorted([f for f in os.listdir(temp_right) if f.endswith('.png')])
        
        if not left_frames or not right_frames:
            raise ValueError(f"No frames found in temporary directories")
        
        print(f"Found {len(left_frames)} frames in left directory and {len(right_frames)} frames in right directory")
        
        # Process each frame pair
        anaglyph_frames = []
        for i, (left_frame, right_frame) in enumerate(tqdm(zip(left_frames, right_frames), total=len(left_frames), desc="Creating anaglyph")):
            # Read frames
            left = cv2.imread(os.path.join(temp_left, left_frame))
            right = cv2.imread(os.path.join(temp_right, right_frame))
            
            if left is None or right is None:
                print(f"Warning: Could not read frame {left_frame} or {right_frame}")
                continue
            
            # Convert BGR to RGB
            left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
            right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
            
            # Create red-cyan anaglyph (red from left, green+blue from right)
            anaglyph = np.zeros_like(left)
            anaglyph[:, :, 0] = left[:, :, 0]   # Red from left
            anaglyph[:, :, 1] = right[:, :, 1]  # Green from right
            anaglyph[:, :, 2] = right[:, :, 2]  # Blue from right
            
            anaglyph_frames.append(anaglyph)
        
        # Save video using imageio
        import imageio
        imageio.mimwrite(output_path, anaglyph_frames, fps=24, quality=7)
        print(f"Red-cyan anaglyph video saved to {output_path}")
        
    finally:
        # Clean up temporary directories
        if os.path.exists(temp_left):
            shutil.rmtree(temp_left)
        if os.path.exists(temp_right):
            shutil.rmtree(temp_right)
        if os.path.exists(temp_anaglyph):
            shutil.rmtree(temp_anaglyph)

def create_stereo_pair(left_dir, right_dir, output_path):
    """Create side-by-side stereo pair video"""
    # Create temporary directories for frames
    temp_left = os.path.join(os.path.dirname(output_path), 'temp_left')
    temp_right = os.path.join(os.path.dirname(output_path), 'temp_right')
    temp_stereo = os.path.join(os.path.dirname(output_path), 'temp_stereo')
    os.makedirs(temp_left, exist_ok=True)
    os.makedirs(temp_right, exist_ok=True)
    os.makedirs(temp_stereo, exist_ok=True)
    
    try:
        # First check for individual frames
        left_frames = sorted([f for f in os.listdir(left_dir) if f.startswith('left_') and f.endswith('.png')])
        right_frames = sorted([f for f in os.listdir(right_dir) if f.startswith('right_') and f.endswith('.png')])
        
        if not left_frames and not right_frames:
            # If no individual frames found, check for videos
            left_videos = sorted([f for f in os.listdir(left_dir) if f.endswith('.mp4')])
            right_videos = sorted([f for f in os.listdir(right_dir) if f.endswith('.mp4')])
            
            if not left_videos or not right_videos:
                raise ValueError(f"No video files or frames found in {left_dir} or {right_dir}")
            
            # Extract frames from all videos
            for video in left_videos:
                video_path = os.path.join(left_dir, video)
                extract_frames_from_video(video_path, temp_left)
            
            for video in right_videos:
                video_path = os.path.join(right_dir, video)
                extract_frames_from_video(video_path, temp_right)
            
            # Get frame lists from extracted videos
            left_frames = sorted([f for f in os.listdir(temp_left) if f.endswith('.png')])
            right_frames = sorted([f for f in os.listdir(temp_right) if f.endswith('.png')])
        else:
            # Copy individual frames to temp directories
            for frame in left_frames:
                shutil.copy(os.path.join(left_dir, frame), os.path.join(temp_left, frame))
            for frame in right_frames:
                shutil.copy(os.path.join(right_dir, frame), os.path.join(temp_right, frame))
        
        if not left_frames or not right_frames:
            raise ValueError(f"No frames found in temporary directories")
        
        print(f"Found {len(left_frames)} frames in left directory and {len(right_frames)} frames in right directory")
        
        # Process each frame pair
        stereo_frames = []
        for i, (left_frame, right_frame) in enumerate(tqdm(zip(left_frames, right_frames), total=len(left_frames), desc="Creating stereo pair")):
            left = cv2.imread(os.path.join(temp_left, left_frame))
            right = cv2.imread(os.path.join(temp_right, right_frame))
            
            if left is None or right is None:
                print(f"Warning: Could not read frame {left_frame} or {right_frame}")
                continue
            
            # Convert BGR to RGB
            left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
            right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
            
            # Create side-by-side stereo pair
            stereo = np.concatenate((left, right), axis=1)
            stereo_frames.append(stereo)
        
        # Save video using imageio
        import imageio
        imageio.mimwrite(output_path, stereo_frames, fps=24, quality=7)
        print(f"Stereo pair video saved to {output_path}")
        
    finally:
        # Clean up temporary directories
        if os.path.exists(temp_left):
            shutil.rmtree(temp_left)
        if os.path.exists(temp_right):
            shutil.rmtree(temp_right)
        if os.path.exists(temp_stereo):
            shutil.rmtree(temp_stereo)

def cleanup_temp_files(dirs):
    """Clean up temporary files and directories"""
    temp_dirs = [
        os.path.join(dirs['stereo_output'], 'batches'),
        os.path.join(dirs['masks_left'], 'batches'),
        os.path.join(dirs['masks_right'], 'batches')
    ]
    
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

def check_existing_files(dirs):
    """Check for existing stereo pairs and masks"""
    left_pairs = sorted([f for f in os.listdir(dirs['stereo_output']) if f.startswith('left_') and f.endswith('.png')])
    right_pairs = sorted([f for f in os.listdir(dirs['stereo_output']) if f.startswith('right_') and f.endswith('.png')])
    left_masks = sorted([f for f in os.listdir(dirs['masks_left']) if f.startswith('mask_') and f.endswith('.png')])
    right_masks = sorted([f for f in os.listdir(dirs['masks_right']) if f.startswith('mask_') and f.endswith('.png')])
    
    return {
        'has_pairs': len(left_pairs) > 0 and len(right_pairs) > 0,
        'has_masks': len(left_masks) > 0 and len(right_masks) > 0,
        'num_pairs': len(left_pairs),
        'num_masks': len(left_masks)
    }

def read_depth_file(depth_path):
    """Read depth data from either .npy or .exr file"""
    if depth_path.endswith('.npy'):
        return np.load(depth_path)
    elif depth_path.endswith('.exr'):
        import OpenEXR
        import Imath
        
        exr_file = OpenEXR.InputFile(depth_path)
        header = exr_file.header()
        dw = header['dataWindow']
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        
        depth_str = exr_file.channel('Z')
        depth = np.frombuffer(depth_str, dtype=np.float32)
        depth = depth.reshape(size[1], size[0])
        return depth
    else:
        raise ValueError(f"Unsupported depth file format: {depth_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert 2D video to 3D using depth maps')
    
    # Basic parameters
    parser.add_argument('--video', required=True, help='Path to input 2D video')
    parser.add_argument('--depth_folder', required=True, help='Path to depth maps folder or single .npy/.exr file')
    parser.add_argument('--output_dir', required=True, help='Base output directory')
    parser.add_argument('--shift_amount', type=int, default=50, help='3D effect strength (10-100)')
    parser.add_argument('--batch_size', type=int, default=60, help='Number of frames per batch')
    
    # Mask parameters
    parser.add_argument('--mask_dilation_kernel', type=int, default=1, 
                       help='Kernel size for mask dilation (1-5)')
    parser.add_argument('--mask_dilation_iterations', type=int, default=1, 
                       help='Number of dilation iterations (1-5)')
    
    # ProPainter parameters
    parser.add_argument('--neighbor_length', type=int, default=10, 
                       help='Number of neighbor frames to consider')
    parser.add_argument('--ref_stride', type=int, default=10, 
                       help='Stride for reference frames')
    parser.add_argument('--raft_iter', type=int, default=20, 
                       help='Number of RAFT iterations')
    parser.add_argument('--save_fps', type=int, default=24, 
                       help='Output video FPS')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.output_dir)
    
    try:
        # Validate inputs and check depth format
        single_npy, depth_path = validate_inputs(args.video, args.depth_folder)
        
        # Create directories
        dirs = create_directories(args.output_dir)
        
        # Check for existing files
        existing_files = check_existing_files(dirs)
        
        if not existing_files['has_pairs']:
            # Extract frames from video
            logger.info("Extracting frames from video...")
            total_frames = extract_frames(args.video, dirs['stereo_output'])
            
            # Generate stereo pairs
            logger.info("Generating stereo pairs...")
            
            if single_npy:
                # Load single .npy file
                all_depths = np.load(depth_path)
                if len(all_depths.shape) != 3:
                    raise ValueError("Single .npy file must contain 3D array of depth maps")
                if all_depths.shape[0] != total_frames:
                    raise ValueError(f"Number of depth maps ({all_depths.shape[0]}) does not match number of frames ({total_frames})")
            else:
                # Get individual depth files
                depth_files = sorted([f for f in os.listdir(depth_path) if f.endswith(('.npy', '.exr'))])
                if len(depth_files) != total_frames:
                    raise ValueError(f"Number of depth maps ({len(depth_files)}) does not match number of frames ({total_frames})")
            
            for i in range(total_frames):
                frame_path = os.path.join(dirs['stereo_output'], f'frame_{i:04d}.png')
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                if single_npy:
                    depth = all_depths[i]
                else:
                    depth_file = os.path.join(depth_path, depth_files[i])
                    depth = read_depth_file(depth_file)
                
                left_img = shift_image_with_transparency(frame, depth, args.shift_amount, 'left')
                right_img = shift_image_with_transparency(frame, depth, args.shift_amount, 'right')
                
                cv2.imwrite(os.path.join(dirs['stereo_output'], f'right_{i:04d}.png'), cv2.cvtColor(left_img, cv2.COLOR_RGBA2BGRA))
                cv2.imwrite(os.path.join(dirs['stereo_output'], f'left_{i:04d}.png'), cv2.cvtColor(right_img, cv2.COLOR_RGBA2BGRA))
                
                if i % 10 == 0:
                    logger.info(f"Processed {i}/{total_frames} frames")
        else:
            logger.info(f"Found existing stereo pairs: {existing_files['num_pairs']} pairs")
            total_frames = existing_files['num_pairs']
        
        if not existing_files['has_masks']:
            # Create masks
            logger.info("Creating masks...")
            create_masks_from_transparent(
                dirs['stereo_output'], 
                dirs['masks_left'],
                args.mask_dilation_kernel,
                args.mask_dilation_iterations
            )
            create_masks_from_transparent(
                dirs['stereo_output'], 
                dirs['masks_right'],
                args.mask_dilation_kernel,
                args.mask_dilation_iterations
            )
        else:
            logger.info(f"Found existing masks: {existing_files['num_masks']} masks")
        
        # ProPainter parameters
        propainter_params = {
            'neighbor_length': args.neighbor_length,
            'ref_stride': args.ref_stride,
            'save_fps': args.save_fps,
            'raft_iter': args.raft_iter
        }
        
        # Calculate number of batches based on actual frame count
        num_batches = (total_frames + args.batch_size - 1) // args.batch_size
        
        # Check if last batch would be too small (less than 2 frames)
        last_batch_size = total_frames % args.batch_size
        if last_batch_size > 0 and last_batch_size < 2:
            num_batches -= 1  # Reduce number of batches to merge small last batch
        
        logger.info(f"Processing {num_batches} batches for {total_frames} frames...")
        
        # Process batches for left and right views separately
        logger.info(f"Processing {num_batches} batches for left view...")
        for batch_num in range(num_batches):
            logger.info(f"Processing batch {batch_num + 1}/{num_batches} for left view...")
            process_batch(batch_num, dirs['stereo_output'], dirs['masks_left'], 
                         dirs['results_left'], args.batch_size, propainter_params, 'left')
        
        logger.info(f"Processing {num_batches} batches for right view...")
        for batch_num in range(num_batches):
            logger.info(f"Processing batch {batch_num + 1}/{num_batches} for right view...")
            process_batch(batch_num, dirs['stereo_output'], dirs['masks_right'], 
                         dirs['results_right'], args.batch_size, propainter_params, 'right')
        
        # Combine batches
        logger.info("Combining batches...")
        combine_batches(dirs['results_left'], dirs['combined_left'])
        combine_batches(dirs['results_right'], dirs['combined_right'])
        
        # Create final outputs
        logger.info("Creating final outputs...")
        create_anaglyph(
            dirs['combined_right'],
            dirs['combined_left'],
            os.path.join(args.output_dir, 'anaglyph_output.mp4')
        )
        create_stereo_pair(
            dirs['combined_right'],
            dirs['combined_left'],
            os.path.join(args.output_dir, 'stereo_pair_output.mp4')
        )
        
        # Cleanup
        cleanup_temp_files(dirs)
        
        logger.info("Process completed successfully!")
        
    except Exception as e:
        logger.error(f"Process failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 