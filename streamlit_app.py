import streamlit as st
import os
import subprocess
from pathlib import Path
import shutil
import logging
from datetime import datetime
import time
import re
import shlex
import mimetypes
import tempfile
import cv2
import numpy as np
import imageio
import zipfile
from auth import Auth

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for paths
WORKSPACE_ROOT = Path("/media/ptgml/ptg1/Bharath_G/High Quality Test/DepthCrafter")
SCRIPT_PATH = WORKSPACE_ROOT / "2d-3d-pipeline" / "integrated_3d_conversion.py"
PYTHON_PATH = "/home/ptgml/anaconda3/envs/depthcrafter/bin/python"

def setup_environment():
    """Ensure all required directories exist"""
    # Create timestamp for this session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Base directories
    base_dir = WORKSPACE_ROOT / "outputs"
    session_dir = base_dir / timestamp
    
    # Create directory structure
    dirs = {
        'base': base_dir,
        'session': session_dir,
        'depthcrafter_output': session_dir / "depthcrafter_output",
        '3d_output': session_dir / "3d_output",
        'temp': session_dir / "temp"
    }
    
    # Create all directories
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs, timestamp

def cleanup_environment(dirs, keep_intermediate=False):
    """Clean up temporary files and optionally intermediate files"""
    try:
        # Always clean temp directory
        if dirs['temp'].exists():
            shutil.rmtree(dirs['temp'])
            logger.info(f"Cleaned up temp directory: {dirs['temp']}")
        
        if not keep_intermediate:
            # Clean intermediate files but keep the resized video and EXR files
            for item in dirs['depthcrafter_output'].glob('*'):
                if not (item.name.endswith('_input.mp4') or 
                       item.name.endswith('.exr')):
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
            # Clean 3d_output subfolders
            for sub in [
                'combined_left', 'combined_right', 'results_left', 'results_right',
                'masks_left', 'masks_right', 'stereo_output']:
                sub_path = dirs['3d_output'] / sub
                if sub_path.exists():
                    shutil.rmtree(sub_path)
            logger.info("Cleaned up intermediate files")
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")

def run_depthcrafter(video_path, output_dir, params):
    """Run DepthCrafter to generate depth maps"""
    cmd = f"""
    python run.py \
        --video-path "{video_path}" \
        --save-folder "{output_dir}" \
        --save-exr \
        --num-inference-steps {params['num_inference_steps']} \
        --guidance-scale {params['guidance_scale']} \
        --window-size {params['window_size']} \
        --max-res {params['max_res']} \
        --cpu-offload sequential
    """
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

def run_3d_conversion(video_path, depth_dir, output_dir, params):
    """Run 2D-3D conversion pipeline using DepthCrafter's resized video and EXR files"""
    # Find the resized input video
    resized_video = next(Path(depth_dir).glob("*_input.mp4"))
    
    # Get the full timestamp from the video filename (support both video and image-as-video)
    match = re.search(r'input_(?:video|image_as_video)_(\d{8}_\d{6})', resized_video.name)
    if match:
        timestamp = match.group(1)
        # Try both possible subfolder names
        depth_subdir_video = Path(depth_dir) / f"input_video_{timestamp}"
        depth_subdir_image = Path(depth_dir) / f"input_image_as_video_{timestamp}"
        if depth_subdir_video.exists():
            depth_subdir = depth_subdir_video
        elif depth_subdir_image.exists():
            depth_subdir = depth_subdir_image
        else:
            raise FileNotFoundError(f"Depth maps directory not found at: {depth_subdir_video} or {depth_subdir_image}")
    else:
        raise ValueError("Could not extract timestamp from video filename")
    
    # Verify paths exist
    if not SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Script not found at: {SCRIPT_PATH}")
    if not os.path.exists(PYTHON_PATH):
        raise FileNotFoundError(f"Python not found at: {PYTHON_PATH}")
    if not depth_subdir.exists():
        raise FileNotFoundError(f"Depth maps directory not found at: {depth_subdir}")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Change to the script directory
    original_dir = os.getcwd()
    os.chdir(SCRIPT_PATH.parent)
    
    try:
        # Check if scipy is installed before attempting to install requirements
        check_scipy_cmd = f"{shlex.quote(PYTHON_PATH)} -c 'import scipy'"
        try:
            subprocess.run(check_scipy_cmd, shell=True, check=True, capture_output=True)
            print("ProPainter requirements already installed")
        except subprocess.CalledProcessError:
            # Install ProPainter requirements only if scipy is not installed
            propainter_reqs = SCRIPT_PATH.parent / "ProPainter" / "requirements.txt"
            if propainter_reqs.exists():
                install_cmd = f"{shlex.quote(PYTHON_PATH)} -m pip install -r {shlex.quote(str(propainter_reqs))}"
                print(f"Installing ProPainter requirements: {install_cmd}")
                subprocess.run(install_cmd, shell=True, check=True)
        
        # Use absolute paths for all directories
        abs_depth_dir = os.path.abspath(depth_subdir)
        abs_output_dir = os.path.abspath(output_dir)
        
        # Use shlex.quote() for proper shell escaping
        cmd = f"""
        {shlex.quote(PYTHON_PATH)} {shlex.quote(str(SCRIPT_PATH))} \
            --video {shlex.quote(str(resized_video))} \
            --depth_folder {shlex.quote(abs_depth_dir)} \
            --output_dir {shlex.quote(abs_output_dir)} \
            --shift_amount {params['shift_amount']} \
            --batch_size {params['batch_size']} \
            --mask_dilation_kernel {params['mask_dilation']} \
            --neighbor_length {params['neighbor_length']} \
            --ref_stride {params['ref_stride']} \
            --raft_iter {params['raft_iter']}
        """
        print(f"Running command: {cmd}")  # Debug print
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error output: {result.stderr}")  # Debug print
            print(f"Command output: {result.stdout}")  # Debug print
            
    finally:
        # Change back to the original directory
        os.chdir(original_dir)
    
    return result

def create_masks_from_transparent(stereo_dir, output_dir, dilation_kernel=2, iterations=1):
    """Create masks from transparent areas in stereo images"""
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all PNG files from stereo directory
    stereo_dir = Path(stereo_dir)
    image_files = sorted(stereo_dir.glob('*.png'))
    
    for img_path in image_files:
        # Read image with alpha channel
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        
        if img is None:
            logger.warning(f"Could not read image: {img_path}")
            continue
        
        # If image has no alpha channel, create a full mask (all white)
        if img.shape[-1] != 4:
            # Create a white mask (all 255) with the same dimensions as the input image
            mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
        else:
            # Extract alpha channel
            alpha = img[:, :, 3]
            # Create binary mask (0 for transparent, 1 for opaque)
            mask = (alpha > 0).astype(np.uint8) * 255
        
        # Apply dilation if specified
        if dilation_kernel > 0:
            kernel = np.ones((dilation_kernel, dilation_kernel), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=iterations)
        
        # Save mask
        output_path = output_dir / f"{img_path.stem}_mask.png"
        cv2.imwrite(str(output_path), mask)

def create_output_videos(output_dir, is_image=False):
    """Create final output videos from processed frames"""
    output_dir = Path(output_dir)
    
    def create_video_writer(output_path, width, height, fps):
        """Create video writer with fallback codecs"""
        # Try different codecs in order of preference
        codecs = [
            ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),  # Default MP4 codec
            ('XVID', cv2.VideoWriter_fourcc(*'XVID')),  # XVID codec
            ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),  # Motion JPEG
            ('X264', cv2.VideoWriter_fourcc(*'X264')),  # H.264
        ]
        
        for codec_name, codec in codecs:
            try:
                writer = cv2.VideoWriter(
                    str(output_path),
                    codec,
                    fps,
                    (width, height)
                )
                if writer.isOpened():
                    logger.info(f"Successfully created video writer with codec: {codec_name}")
                    return writer
                writer.release()
            except Exception as e:
                logger.warning(f"Failed to create video writer with codec {codec_name}: {str(e)}")
                continue
        
        raise RuntimeError("Failed to create video writer with any available codec")
    
    # Create anaglyph video
    anaglyph_frames = sorted((output_dir / 'anaglyph_output').glob('*.png'))
    if anaglyph_frames:
        first_frame = cv2.imread(str(anaglyph_frames[0]))
        if first_frame is None:
            logger.error("Could not read first anaglyph frame")
            return
            
        height, width = first_frame.shape[:2]
        fps = 1 if is_image else 24
        
        try:
            anaglyph_writer = create_video_writer(
                output_dir / 'anaglyph_output.mp4',
                width, height, fps
            )
            
            for frame_path in anaglyph_frames:
                frame = cv2.imread(str(frame_path))
                if frame is not None:
                    anaglyph_writer.write(frame)
                else:
                    logger.warning(f"Could not read anaglyph frame: {frame_path}")
            
            anaglyph_writer.release()
            logger.info("Successfully created anaglyph output video")
        except Exception as e:
            logger.error(f"Error creating anaglyph video: {str(e)}")
    
    # Create stereo pair video
    stereo_frames = sorted((output_dir / 'stereo_output').glob('*.png'))
    if stereo_frames:
        first_frame = cv2.imread(str(stereo_frames[0]))
        if first_frame is None:
            logger.error("Could not read first stereo frame")
            return
            
        height, width = first_frame.shape[:2]
        fps = 1 if is_image else 24
        
        # Create stereo output directory if it doesn't exist
        stereo_output_dir = output_dir / 'stereo_output'
        stereo_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            stereo_writer = create_video_writer(
                output_dir / 'stereo_pair_output.mp4',
                width, height, fps
            )
            
            frame_count = 0
            for frame_path in stereo_frames:
                frame = cv2.imread(str(frame_path))
                if frame is not None:
                    stereo_writer.write(frame)
                    frame_count += 1
                else:
                    logger.warning(f"Could not read stereo frame: {frame_path}")
            
            stereo_writer.release()
            
            # Verify the output video was created and has frames
            output_path = output_dir / 'stereo_pair_output.mp4'
            if output_path.exists() and frame_count > 0:
                logger.info(f"Successfully created stereo pair output video with {frame_count} frames")
            else:
                logger.error("Failed to create stereo pair output video or no frames were written")
        except Exception as e:
            logger.error(f"Error creating stereo video: {str(e)}")

def main():
    st.set_page_config(
        page_title="2D to 3D Video Converter",
        page_icon="🎥",
        layout="wide"
    )
    
    # Initialize authentication
    auth = Auth()
    
    # Add navigation menu
    if "authenticated" in st.session_state and st.session_state.authenticated:
        # Create a sidebar for navigation
        with st.sidebar:
            st.title("Navigation")
            if st.button("📁 Projects"):
                st.session_state.current_page = "projects"
                st.rerun()
            
            if st.button("➕ New Project"):
                st.session_state.current_page = "new_project"
                st.rerun()
            
            st.markdown("---")
            st.write(f"👤 Logged in as: {st.session_state.username}")
            if st.button("🚪 Logout"):
                # Clear session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
    
    # Check authentication and handle pages
    if not auth.main():
        return  # Stop if not authenticated
    
    # If we get here, we're authenticated and on the conversion page
    st.title("🎥 2D to 3D Video Converter")
    st.markdown("""
    This application converts 2D videos or images to 3D using depth estimation 
    and advanced inpainting techniques for 3D conversion.
    """)

    # Check if we're opening an existing project
    if 'current_project' in st.session_state and st.session_state.current_page == "conversion":
        project = st.session_state.current_project
        st.info(f"📂 Opened Project: {project['project_name']}")
        
        # Display project details
        if 'input_file' in project:
            input_info = project['input_file']
            st.info(f"Input File: {input_info['name']} ({input_info['type']})")
            if 'resolution' in input_info:
                st.info(f"Resolution: {input_info['resolution']}")
            
            # Try to load the input file from the project's output directory
            if 'output_files' in project:
                input_path = None
                # Look for the input file in the depthcrafter output directory
                depth_dir = Path(project['output_files']['depth_maps'])
                for file in depth_dir.glob("*_input.mp4"):
                    input_path = file
                    break
                
                if input_path and input_path.exists():
                    # Store the input file in session state
                    st.session_state.project_input_path = str(input_path)
                    st.success("✅ Input file loaded from project")
                else:
                    st.warning("⚠️ Original input file not found in project directory")

                # Display project results if they exist
                if project.get('processing_info', {}).get('status') == 'completed':
                    st.header("🎬 Project Results")
                    
                    # Create columns for results
                    result_col1, result_col2 = st.columns(2)
                    
                    with result_col1:
                        if "anaglyph" in project['output_files']:
                            st.subheader("Anaglyph (Red-Cyan) 3D")
                            anaglyph_path = Path(project['output_files']['anaglyph'])
                            if anaglyph_path.exists():
                                with open(anaglyph_path, "rb") as video_file:
                                    video_bytes = video_file.read()
                                    st.video(video_bytes, format="video/mp4")
                                    st.download_button(
                                        "📥 Download Anaglyph Video",
                                        video_bytes,
                                        f"anaglyph_output_{project['timestamp']}.mp4",
                                        mime="video/mp4"
                                    )
                            else:
                                st.error(f"Anaglyph video not found at: {anaglyph_path}")
                    
                    with result_col2:
                        if "stereo" in project['output_files']:
                            st.subheader("Side-by-Side 3D")
                            stereo_path = Path(project['output_files']['stereo'])
                            if stereo_path.exists():
                                try:
                                    with open(stereo_path, "rb") as video_file:
                                        video_bytes = video_file.read()
                                        # Use format parameter for better compatibility
                                        st.video(video_bytes, format="video/mp4")
                                        st.download_button(
                                            "📥 Download Side-by-Side Video",
                                            video_bytes,
                                            f"stereo_pair_output_{project['timestamp']}.mp4",
                                            mime="video/mp4"
                                        )
                                except Exception as e:
                                    st.error(f"Error displaying stereo video: {str(e)}")
                                    logger.error(f"Error displaying stereo video: {str(e)}")
                            else:
                                st.error(f"Stereo pair video not found at: {stereo_path}")

    # File Upload (accept both video and image)
    uploaded_file = st.file_uploader("Upload 2D Video or Image", type=['mp4', 'avi', 'mov', 'jpg', 'jpeg', 'png'])

    # If we have a project input file, use it instead of the uploaded file
    if 'project_input_path' in st.session_state and not uploaded_file:
        st.info("Using input file from project")
        # The rest of the code will use st.session_state.project_input_path instead of uploaded_file

    # Model Selection
    st.header("🧠 Depth Estimation Model")
    model_options = [
        "DepthCrafter",
        "VideoDepth (Coming Soon)",
        "RollingDepth (Coming Soon)",
        "DepthPro (Coming Soon)",
        "DistillAnyDepth (Coming Soon)",
        "Marigold (Coming Soon)"
    ]
    selected_model = st.selectbox(
        "Choose a depth estimation model",
        model_options,
        index=0
    )

    if selected_model != "DepthCrafter":
        st.info("This model is coming soon! Please select DepthCrafter to proceed.")
        st.stop()

    # Parameters Section
    st.header("⚙️ Parameters")
    
    # Create two columns for parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Depth Generation Settings")
        # Use project parameters if available
        default_depth_params = st.session_state.get('project_depth_params', {
            'num_inference_steps': 10,
            'guidance_scale': 2.0,
            'window_size': 100,
            'max_res': 1024
        })
        
        depth_params = {
            'num_inference_steps': st.slider("Inference Steps", 1, 20, 
                                           default_depth_params.get('num_inference_steps', 10),
                                           help="Number of denoising steps for depth generation"),
            'guidance_scale': st.slider("Guidance Scale", 0.1, 5.0,
                                      default_depth_params.get('guidance_scale', 2.0),
                                      help="Controls how closely the depth follows the input"),
            'window_size': st.slider("Window Size", 50, 200,
                                   default_depth_params.get('window_size', 100),
                                   help="Size of the processing window for temporal consistency"),
            'max_res': st.selectbox("Max Resolution", [512, 768, 1024],
                                  index=[512, 768, 1024].index(default_depth_params.get('max_res', 1024)),
                                  help="Maximum resolution for processing")
        }
    
    with col2:
        st.subheader("3D Conversion Settings")
        # Use project parameters if available
        default_conversion_params = st.session_state.get('project_conversion_params', {
            'shift_amount': 30,
            'batch_size': 60,
            'mask_dilation': 2,
            'neighbor_length': 10,
            'ref_stride': 10,
            'raft_iter': 20
        })
        
        conversion_params = {
            'shift_amount': st.slider("3D Effect Strength", 10, 60,
                                    default_conversion_params.get('shift_amount', 30),
                                    help="Controls the strength of the 3D effect"),
            'batch_size': st.slider("Batch Size", 30, 120,
                                  default_conversion_params.get('batch_size', 60),
                                  help="Number of frames to process at once"),
            'mask_dilation': st.slider("Mask Dilation", 1, 5,
                                     default_conversion_params.get('mask_dilation', 2),
                                     help="Size of mask dilation kernel"),
            'neighbor_length': st.slider("Neighbor Length", 5, 20,
                                       default_conversion_params.get('neighbor_length', 10),
                                       help="Number of neighbor frames to consider"),
            'ref_stride': st.slider("Reference Stride", 5, 20,
                                  default_conversion_params.get('ref_stride', 10),
                                  help="Stride for reference frames"),
            'raft_iter': st.slider("RAFT Iterations", 10, 30,
                                 default_conversion_params.get('raft_iter', 20),
                                 help="Number of RAFT iterations")
        }
    
    # Output Format Selection
    st.subheader("🎬 Output Format")
    output_format = st.multiselect(
        "Select output formats",
        ["Anaglyph", "Side-by-Side", "Both"],
        default=["Both"],
        help="Choose the format(s) for the output 3D video"
    )
    
    # File Management Options
    st.subheader("💾 Intermediate Asset Management")
    keep_intermediate = st.checkbox(
        "Keep Intermediate Assets (depth maps, masks, etc.)",
        value=False,
        help="If checked, intermediate files will be kept in the output directory"
    )
    
    if st.button("🚀 Start Conversion", type="primary"):
        if uploaded_file is not None or 'project_input_path' in st.session_state:
            # Initialize variables
            width = None
            height = None
            project_config = None
            
            # Setup directories
            dirs, timestamp = setup_environment()
            
            try:
                if 'project_input_path' in st.session_state:
                    # Use the project's input file
                    video_path = Path(st.session_state.project_input_path)
                    is_image = False  # We know it's a video since we saved it as MP4
                    # Get video dimensions
                    cap = cv2.VideoCapture(str(video_path))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    cap.release()
                else:
                    # Handle newly uploaded file
                    mime_type, _ = mimetypes.guess_type(uploaded_file.name)
                    is_image = mime_type and mime_type.startswith('image')
                    is_video = mime_type and mime_type.startswith('video')
                    
                    if is_image:
                        # Read image and duplicate to 10 frames
                        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                        frames = [img] * 10
                        height, width, _ = img.shape
                        temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
                        out = cv2.VideoWriter(temp_video.name, cv2.VideoWriter_fourcc(*'mp4v'), 24, (width, height))
                        for frame in frames:
                            out.write(frame)
                        out.release()
                        video_path = dirs['session'] / f"input_image_as_video_{timestamp}.mp4"
                        shutil.copy(temp_video.name, video_path)
                    elif is_video:
                        # Save uploaded video
                        video_path = dirs['session'] / f"input_video_{timestamp}.mp4"
                        with open(video_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        # Get video dimensions
                        cap = cv2.VideoCapture(str(video_path))
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        cap.release()
                    else:
                        st.error("Unsupported file type.")
                        return

                # Create project configuration
                project_config = {
                    "input_file": {
                        "name": uploaded_file.name if uploaded_file else Path(st.session_state.project_input_path).name,
                        "type": "image" if is_image else "video",
                        "resolution": f"{width}x{height}",
                        "duration": "N/A" if is_image else "video duration"
                    },
                    "parameters": {
                        "depth_params": depth_params,
                        "conversion_params": conversion_params
                    },
                    "output_files": {
                        "anaglyph": str(dirs['3d_output'] / "anaglyph_output.mp4"),
                        "stereo": str(dirs['3d_output'] / "stereo_pair_output.mp4"),
                        "depth_maps": str(dirs['depthcrafter_output']),
                        "masks": str(dirs['3d_output'] / "masks_left"),
                        "intermediate": str(dirs['3d_output'])
                    },
                    "processing_info": {
                        "start_time": datetime.now().isoformat(),
                        "status": "processing"
                    }
                }
                
                # Save project configuration
                if 'current_project' in st.session_state and 'project_id' in st.session_state.current_project:
                    # Update existing project
                    project_id = st.session_state.current_project['project_id']
                    auth.update_project_config(project_id, project_config)
                else:
                    # Create new project
                    project_id = auth.save_project_config(
                        st.session_state.username,
                        st.session_state.current_project["project_name"],
                        project_config
                    )
                    # Update session state with project ID
                    st.session_state.current_project["project_id"] = project_id
                
                # Progress tracking
                status_steps = [
                    "Extracting frames...",
                    "Estimating depths...",
                    "Creating stereo/anaglyph views...",
                    "Creating masks...",
                    "Painting occlusions...",
                    "Creating results..."
                ]
                progress_bar = st.progress(0)
                status_text = st.empty()
                step_count = len(status_steps)
                
                # Step 1: Extracting frames
                status_text.text(status_steps[0])
                progress_bar.progress(1/step_count)
                
                # Step 2: Estimating depths
                status_text.text(status_steps[1])
                result = run_depthcrafter(str(video_path), str(dirs['depthcrafter_output']), depth_params)
                if result.returncode != 0:
                    st.error(f"Error in depth generation: {result.stderr}")
                    return
                progress_bar.progress(2/step_count)
                
                # Step 3: Creating stereo/anaglyph views
                status_text.text(status_steps[2])
                result = run_3d_conversion(
                    str(video_path),
                    str(dirs['depthcrafter_output']),
                    str(dirs['3d_output']),
                    conversion_params
                )
                if result.returncode != 0:
                    st.error(f"Error in 3D conversion: {result.stderr}")
                    return
                progress_bar.progress(3/step_count)
                
                # Step 4: Creating masks
                status_text.text(status_steps[3])
                # Create masks for left and right views
                create_masks_from_transparent(
                    dirs['3d_output'] / 'stereo_output',
                    dirs['3d_output'] / 'masks_left',
                    conversion_params['mask_dilation'],
                    1  # iterations
                )
                create_masks_from_transparent(
                    dirs['3d_output'] / 'stereo_output',
                    dirs['3d_output'] / 'masks_right',
                    conversion_params['mask_dilation'],
                    1  # iterations
                )
                progress_bar.progress(4/step_count)
                
                # Step 5: Painting occlusions
                status_text.text(status_steps[4])
                # The occlusion painting is handled within run_3d_conversion
                progress_bar.progress(5/step_count)
                
                # Step 6: Creating results
                status_text.text(status_steps[5])
                # Create final output videos
                create_output_videos(dirs['3d_output'], is_image)
                progress_bar.progress(1.0)
                status_text.text("Conversion complete! 🎉")
                
                # Display results
                st.header("🎬 Results")
                
                # Update project configuration with completion status
                project_config["processing_info"]["end_time"] = datetime.now().isoformat()
                project_config["processing_info"]["status"] = "completed"
                auth.update_project_config(project_id, project_config)
                
                # Store results in session state for persistence after download
                st.session_state['last_results'] = {
                    'is_image': is_image,
                    'timestamp': timestamp,
                    'dirs': dirs,
                    'project_id': project_id
                }
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                logger.error(f"Error during processing: {str(e)}")
                
                # Only update project config if it was created
                if project_config is not None:
                    project_config["processing_info"]["end_time"] = datetime.now().isoformat()
                    project_config["processing_info"]["status"] = "failed"
                    project_config["processing_info"]["error"] = str(e)
                    auth.update_project_config(project_id, project_config)
            
            finally:
                # Cleanup based on user preference
                cleanup_environment(dirs, keep_intermediate)
        else:
            st.warning("Please upload a video or image file first!")

    # After processing, or on rerun, show results if present
    if 'last_results' in st.session_state:
        is_image = st.session_state['last_results']['is_image']
        timestamp = st.session_state['last_results']['timestamp']
        dirs = st.session_state['last_results']['dirs']
        output_format = st.session_state.get('output_format', ["Both"])
        keep_intermediate = st.session_state.get('keep_intermediate', False)
        result_col1, result_col2 = st.columns(2)
        with result_col1:
            if "Anaglyph" in output_format or "Both" in output_format:
                st.subheader("Anaglyph (Red-Cyan) 3D")
                anaglyph_path = dirs['3d_output'] / "anaglyph_output.mp4"
                if anaglyph_path.exists():
                    if is_image:
                        st.markdown("**Single-frame Anaglyph Preview**")
                        reader = imageio.get_reader(anaglyph_path)
                        frame = reader.get_data(0)
                        st.image(frame, caption="Anaglyph 3D Frame", channels="RGB")
                        reader.close()
                        import io
                        from PIL import Image
                        buf = io.BytesIO()
                        Image.fromarray(frame).save(buf, format='PNG')
                        st.download_button(
                            "📥 Download Anaglyph 3D Image",
                            buf.getvalue(),
                            f"anaglyph_output_{timestamp}.png",
                            mime="image/png"
                        )
                    else:
                        with open(anaglyph_path, "rb") as video_file:
                            video_bytes = video_file.read()
                            st.video(video_bytes, format="video/mp4")
                            st.download_button(
                                "📥 Download Anaglyph Video",
                                video_bytes,
                                f"anaglyph_output_{timestamp}.mp4",
                                mime="video/mp4"
                            )
                else:
                    st.error(f"Anaglyph video not found at: {anaglyph_path}")
        with result_col2:
            if "Side-by-Side" in output_format or "Both" in output_format:
                st.subheader("Side-by-Side 3D")
                stereo_path = dirs['3d_output'] / "stereo_pair_output.mp4"
                if stereo_path.exists():
                    if is_image:
                        st.markdown("**Single-frame Stereo Preview**")
                        reader = imageio.get_reader(stereo_path)
                        frame = reader.get_data(0)
                        st.image(frame, caption="Stereo 3D Frame", channels="RGB")
                        reader.close()
                        import io
                        from PIL import Image
                        buf = io.BytesIO()
                        Image.fromarray(frame).save(buf, format='PNG')
                        st.download_button(
                            "📥 Download Stereo 3D Image",
                            buf.getvalue(),
                            f"stereo_pair_output_{timestamp}.png",
                            mime="image/png"
                        )
                    else:
                        with open(stereo_path, "rb") as video_file:
                            video_bytes = video_file.read()
                            st.video(video_bytes)
                            st.download_button(
                                "📥 Download Side-by-Side Video",
                                video_bytes,
                                f"stereo_pair_output_{timestamp}.mp4",
                                mime="video/mp4"
                            )
                else:
                    st.error(f"Stereo pair video not found at: {stereo_path}")
        
        # Add download zip button for intermediate assets only if they are kept
        if keep_intermediate:
            st.subheader("Download Intermediate Assets")
            import io
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add depth maps
                depth_dir = dirs['depthcrafter_output']
                if depth_dir.exists():
                    for root, _, files in os.walk(depth_dir):
                        for file in files:
                            if file.endswith('.exr') or file.endswith('.npy'):
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, start=depth_dir)
                                zipf.write(file_path, arcname=f"depth_maps/{arcname}")
                
                # Add masks
                for mask_dir in ['masks_left', 'masks_right']:
                    mask_path = dirs['3d_output'] / mask_dir
                    if mask_path.exists():
                        for root, _, files in os.walk(mask_path):
                            for file in files:
                                if file.endswith('.png'):
                                    file_path = os.path.join(root, file)
                                    arcname = os.path.relpath(file_path, start=dirs['3d_output'])
                                    zipf.write(file_path, arcname=f"masks/{arcname}")
                
                # Add warped frames (stereo_output)
                stereo_output_path = dirs['3d_output'] / 'stereo_output'
                if stereo_output_path.exists():
                    for root, _, files in os.walk(stereo_output_path):
                        for file in files:
                            if file.endswith('.png'):
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, start=dirs['3d_output'])
                                zipf.write(file_path, arcname=f"warped_frames/{arcname}")
            
            # Only show download button if there are files in the zip
            if zipf.namelist():
                zip_buf.seek(0)
                st.download_button(
                    "📦 Download Depth Maps, Masks, Warped Frames (zip)",
                    zip_buf.getvalue(),
                    f"intermediate_assets_{timestamp}.zip",
                    mime="application/zip"
                )
            else:
                st.warning("No intermediate assets found to download.")

if __name__ == "__main__":
    main() 