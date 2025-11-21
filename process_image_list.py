#!/usr/bin/env python3
"""
Script to process a list of images with SAM 3D Body.
Based on sections 1-5 of the demo_human.ipynb notebook.

This script processes multiple images and saves results organized by type:
- meshes/ : 3D mesh files (PLY format)
- overlays/ : Mesh overlay images (mesh overlaid on original image)
- bboxes/ : Bounding box images
- skeletons/ : 2D skeleton visualizations (keypoints + bbox)
- focal_lengths/ : Focal length JSON files
- skeletons_video.mp4 : Video of skeleton visualizations
- overlays_video.mp4 : Video of mesh overlay visualizations
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add notebook utils to path if running from root
notebook_dir = Path(__file__).parent / "notebook"
if notebook_dir.exists():
    sys.path.insert(0, str(notebook_dir))

from notebook.utils import (
    setup_sam_3d_body,
    setup_visualizer,
    visualize_2d_results,
    save_mesh_results,
)


def find_images_in_directory(directory_path: str, recursive: bool = True) -> List[str]:
    """
    Find all .jpg and .png images in a directory.
    
    Args:
        directory_path: Path to directory containing images
        recursive: If True, search recursively in subdirectories
        
    Returns:
        List of image paths (sorted)
    """
    if not os.path.isdir(directory_path):
        raise ValueError(f"Directory not found: {directory_path}")
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_paths = []
    
    if recursive:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if any(file.lower().endswith(ext.lower()) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
    else:
        for file in os.listdir(directory_path):
            if any(file.lower().endswith(ext.lower()) for ext in image_extensions):
                image_paths.append(os.path.join(directory_path, file))
    
    return sorted(image_paths)


def save_results_by_type(
    img_cv2: np.ndarray,
    outputs: List[dict],
    faces: np.ndarray,
    visualizer,
    output_dir: str,
    image_name: str,
) -> dict:
    """
    Save results organized by type in separate folders.
    
    Args:
        img_cv2: Input image (BGR format)
        outputs: List of person outputs from SAM 3D Body
        faces: Mesh faces from estimator
        visualizer: Skeleton visualizer
        output_dir: Base output directory
        image_name: Base name for output files
        
    Returns:
        Dictionary with counts of saved files
    """
    # Create subdirectories for each output type
    dirs = {
        'meshes': os.path.join(output_dir, 'meshes'),
        'overlays': os.path.join(output_dir, 'overlays'),
        'bboxes': os.path.join(output_dir, 'bboxes'),
        'skeletons': os.path.join(output_dir, 'skeletons'),
        'focal_lengths': os.path.join(output_dir, 'focal_lengths'),
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    saved_counts = {
        'meshes': 0,
        'overlays': 0,
        'bboxes': 0,
        'skeletons': 0,
        'focal_lengths': 0,
    }
    
    # Save focal length (once per image, from first person)
    if outputs:
        focal_length_data = {"focal_length": float(outputs[0]["focal_length"])}
        focal_length_path = os.path.join(dirs['focal_lengths'], f"{image_name}_focal_length.json")
        with open(focal_length_path, "w") as f:
            json.dump(focal_length_data, f, indent=2)
        saved_counts['focal_lengths'] = 1
    
    # Visualize 2D results with skeleton
    skeleton_images = visualize_2d_results(img_cv2, outputs, visualizer)
    
    # Import renderer for mesh visualization
    from sam_3d_body.visualization.renderer import Renderer
    
    # LIGHT_BLUE color constant
    LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
    
    for pid, person_output in enumerate(outputs):
        # Create renderer for this person
        renderer = Renderer(focal_length=person_output["focal_length"], faces=faces)
        
        # Save PLY mesh
        tmesh = renderer.vertices_to_trimesh(
            person_output["pred_vertices"], person_output["pred_cam_t"], LIGHT_BLUE
        )
        mesh_filename = f"{image_name}_mesh_{pid:03d}.ply"
        mesh_path = os.path.join(dirs['meshes'], mesh_filename)
        tmesh.export(mesh_path)
        saved_counts['meshes'] += 1
        
        # Save overlay image
        img_mesh_overlay = (
            renderer(
                person_output["pred_vertices"],
                person_output["pred_cam_t"],
                img_cv2.copy(),
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
            )
            * 255
        ).astype(np.uint8)
        overlay_filename = f"{image_name}_overlay_{pid:03d}.png"
        overlay_path = os.path.join(dirs['overlays'], overlay_filename)
        cv2.imwrite(overlay_path, img_mesh_overlay)
        saved_counts['overlays'] += 1
        
        # Save bbox image (from skeleton visualization, but we'll create a simpler version)
        img_bbox = img_cv2.copy()
        bbox = person_output["bbox"]
        img_bbox = cv2.rectangle(
            img_bbox,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 255, 0),
            4,
        )
        bbox_filename = f"{image_name}_bbox_{pid:03d}.png"
        bbox_path = os.path.join(dirs['bboxes'], bbox_filename)
        cv2.imwrite(bbox_path, img_bbox)
        saved_counts['bboxes'] += 1
        
        # Save skeleton visualization (with keypoints and bbox)
        skeleton_filename = f"{image_name}_skeleton_{pid:03d}.png"
        skeleton_path = os.path.join(dirs['skeletons'], skeleton_filename)
        cv2.imwrite(skeleton_path, skeleton_images[pid])
        saved_counts['skeletons'] += 1
    
    return saved_counts


def create_video_from_images(
    image_dir: str,
    output_video_path: str,
    fps: float = 30.0,
    pattern: str = "*",
) -> bool:
    """
    Create an MP4 video from images in a directory.
    
    Args:
        image_dir: Directory containing images
        output_video_path: Path to save the output video
        fps: Frames per second for the video
        pattern: Filename pattern to match (e.g., "*_skeleton_*.png")
        
    Returns:
        True if video was created successfully, False otherwise
    """
    import glob
    
    # Find all matching images
    image_pattern = os.path.join(image_dir, pattern)
    image_files = sorted(glob.glob(image_pattern))
    
    if not image_files:
        print(f"Warning: No images found matching pattern {pattern} in {image_dir}")
        return False
    
    # Read first image to get dimensions
    first_img = cv2.imread(image_files[0])
    if first_img is None:
        print(f"Error: Could not read first image: {image_files[0]}")
        return False
    
    height, width = first_img.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not open video writer for {output_video_path}")
        return False
    
    # Write all frames
    for img_path in tqdm(image_files, desc=f"Creating video: {os.path.basename(output_video_path)}"):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image: {img_path}")
            continue
        
        # Resize if dimensions don't match
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        
        out.write(img)
    
    out.release()
    print(f"✓ Created video: {output_video_path} ({len(image_files)} frames)")
    return True


def create_videos_from_outputs(output_dir: str, fps: float = 30.0) -> dict:
    """
    Create MP4 videos from skeleton and overlay images.
    
    Args:
        output_dir: Base output directory containing subdirectories
        fps: Frames per second for the videos
        
    Returns:
        Dictionary with paths to created videos
    """
    videos_created = {}
    
    skeleton_dir = os.path.join(output_dir, 'skeletons')
    overlay_dir = os.path.join(output_dir, 'overlays')
    
    # Create skeleton video
    if os.path.exists(skeleton_dir):
        skeleton_video_path = os.path.join(output_dir, 'skeletons_video.mp4')
        if create_video_from_images(skeleton_dir, skeleton_video_path, fps, "*_skeleton_*.png"):
            videos_created['skeletons'] = skeleton_video_path
    
    # Create overlay video
    if os.path.exists(overlay_dir):
        overlay_video_path = os.path.join(output_dir, 'overlays_video.mp4')
        if create_video_from_images(overlay_dir, overlay_video_path, fps, "*_overlay_*.png"):
            videos_created['overlays'] = overlay_video_path
    
    return videos_created


def load_image_list(image_list_path: str) -> List[str]:
    """
    Load image paths from a text file (one path per line).
    
    Args:
        image_list_path: Path to text file containing image paths
        
    Returns:
        List of image paths
    """
    image_paths = []
    with open(image_list_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                image_paths.append(line)
    return image_paths


def process_images(
    image_paths: List[str],
    output_dir: str,
    hf_repo_id: str = "facebook/sam-3d-body-dinov3",
    detector_name: str = "vitdet",
    segmentor_name: str = "sam2",
    fov_name: str = "moge2",
    device: str = "cuda",
    detector_path: str = "",
    segmentor_path: str = "",
    fov_path: str = "",
    create_videos: bool = True,
    video_fps: float = 30.0,
):
    """
    Process a list of images with SAM 3D Body and save results.
    
    Args:
        image_paths: List of paths to input images
        output_dir: Directory to save output files
        hf_repo_id: HuggingFace repository ID for the model
        detector_name: Name of detector to use
        segmentor_name: Name of segmentor to use
        fov_name: Name of FOV estimator to use
        device: Device to use (cuda/cpu)
        detector_path: Path to detector model (optional)
        segmentor_path: Path to segmentor model (optional)
        fov_path: Path to FOV estimator model (optional)
    """
    # Set up SAM 3D Body estimator
    print("Setting up SAM 3D Body estimator...")
    estimator = setup_sam_3d_body(
        hf_repo_id=hf_repo_id,
        detector_name=detector_name,
        segmentor_name=segmentor_name,
        fov_name=fov_name,
        detector_path=detector_path,
        segmentor_path=segmentor_path,
        fov_path=fov_path,
        device=device,
    )
    
    # Set up visualizer for skeleton visualization
    print("Setting up visualizer...")
    visualizer = setup_visualizer()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    successful = 0
    failed = 0
    
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            # Check if image exists
            if not os.path.exists(image_path):
                print(f"\nWarning: Image not found: {image_path}")
                failed += 1
                continue
            
            # Load image to verify it's valid
            img_cv2 = cv2.imread(image_path)
            if img_cv2 is None:
                print(f"\nWarning: Could not load image: {image_path}")
                failed += 1
                continue
            
            # Process the image with SAM 3D Body
            outputs = estimator.process_one_image(image_path)
            
            if not outputs:
                print(f"\nWarning: No people detected in {image_path}")
                failed += 1
                continue
            
            # Get image name without extension
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            
            # Save all results organized by type
            saved_counts = save_results_by_type(
                img_cv2, outputs, estimator.faces, visualizer, output_dir, image_name
            )
            
            print(f"\n✓ Processed {image_path}")
            print(f"  Number of people detected: {len(outputs)}")
            print(f"  Saved files:")
            print(f"    - Meshes: {saved_counts['meshes']}")
            print(f"    - Overlays: {saved_counts['overlays']}")
            print(f"    - BBoxes: {saved_counts['bboxes']}")
            print(f"    - Skeletons: {saved_counts['skeletons']}")
            print(f"    - Focal lengths: {saved_counts['focal_lengths']}")
            
            successful += 1
            
        except Exception as e:
            print(f"\nError processing {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    # Create videos from skeleton and overlay images
    videos_created = {}
    if create_videos and successful > 0:
        print("\n" + "="*60)
        print("Creating videos from processed images...")
        print("="*60)
        videos_created = create_videos_from_outputs(output_dir, fps=video_fps)
    
    # Print summary
    print("\n" + "="*60)
    print("Processing Summary:")
    print(f"  Total images: {len(image_paths)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_dir}")
    if videos_created:
        print(f"\n  Videos created:")
        for video_type, video_path in videos_created.items():
            print(f"    - {video_type}: {video_path}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Process images with SAM 3D Body",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all .jpg and .png images in a directory (recursive)
  python process_image_list.py --image_dir /path/to/images --output_dir ./output

  # Process images from a list file
  python process_image_list.py --image_list images.txt --output_dir ./output

  # Process a single image
  python process_image_list.py --images image1.jpg --output_dir ./output

  # Process multiple images
  python process_image_list.py --images img1.jpg img2.jpg img3.jpg --output_dir ./output

  # Use different model
  python process_image_list.py --image_dir /path/to/images --output_dir ./output \\
      --hf_repo_id facebook/sam-3d-body-vith

Image list file format (one path per line):
  /path/to/image1.jpg
  /path/to/image2.png
  # This is a comment
  /path/to/image3.jpg
        """,
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image_dir",
        type=str,
        help="Directory containing images (.jpg and .png files will be found automatically)",
    )
    input_group.add_argument(
        "--image_list",
        type=str,
        help="Path to text file containing list of image paths (one per line)",
    )
    input_group.add_argument(
        "--images",
        nargs="+",
        help="One or more image paths to process",
    )
    parser.add_argument(
        "--no_recursive",
        action="store_true",
        help="If --image_dir is used, only search in the top-level directory (not subdirectories)",
    )
    
    # Output options
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save output files (default: ./output)",
    )
    
    # Model options
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        default="facebook/sam-3d-body-dinov3",
        help="HuggingFace repository ID for the model (default: facebook/sam-3d-body-dinov3)",
    )
    parser.add_argument(
        "--detector_name",
        type=str,
        default="vitdet",
        help="Human detection model name (default: vitdet)",
    )
    parser.add_argument(
        "--segmentor_name",
        type=str,
        default="sam2",
        help="Human segmentation model name (default: sam2)",
    )
    parser.add_argument(
        "--fov_name",
        type=str,
        default="moge2",
        help="FOV estimation model name (default: moge2)",
    )
    parser.add_argument(
        "--detector_path",
        type=str,
        default="",
        help="Path to human detection model (optional)",
    )
    parser.add_argument(
        "--segmentor_path",
        type=str,
        default="",
        help="Path to human segmentation model (optional)",
    )
    parser.add_argument(
        "--fov_path",
        type=str,
        default="",
        help="Path to FOV estimation model (optional)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--no_videos",
        action="store_true",
        help="Skip video generation (default: videos are created)",
    )
    parser.add_argument(
        "--video_fps",
        type=float,
        default=30.0,
        help="Frames per second for output videos (default: 30.0)",
    )
    
    args = parser.parse_args()
    
    # Get image paths
    if args.image_dir:
        try:
            image_paths = find_images_in_directory(args.image_dir, recursive=not args.no_recursive)
            if not image_paths:
                print(f"Error: No .jpg or .png images found in directory: {args.image_dir}")
                sys.exit(1)
            print(f"Found {len(image_paths)} images in {args.image_dir}")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    elif args.image_list:
        if not os.path.exists(args.image_list):
            print(f"Error: Image list file not found: {args.image_list}")
            sys.exit(1)
        image_paths = load_image_list(args.image_list)
    else:
        image_paths = args.images
    
    if not image_paths:
        print("Error: No images to process")
        sys.exit(1)
    
    # Process images
    process_images(
        image_paths=image_paths,
        output_dir=args.output_dir,
        hf_repo_id=args.hf_repo_id,
        detector_name=args.detector_name,
        segmentor_name=args.segmentor_name,
        fov_name=args.fov_name,
        device=args.device,
        detector_path=args.detector_path,
        segmentor_path=args.segmentor_path,
        fov_path=args.fov_path,
        create_videos=not args.no_videos,
        video_fps=args.video_fps,
    )


if __name__ == "__main__":
    main()

