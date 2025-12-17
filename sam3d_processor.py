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
import os
import copy
import yaml
import sys
from pathlib import Path
from typing import List
import cv2
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

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

from we_cfg.fdc.config import RectificationConfig, StereoDataConfig
from we_cfg.fdc.rectify_video import VideoRectifier, load_rectifier_from_cfg
from we_cfg.fdc.data_define import StereoData, Kpt3D, BatchKpt3D, HumanPose
from we_cfg.fdc.data_loader import StereoDataLoader, load_stereo_data_loader_from_cfg   
from we_cfg.fdc.config import HumanReconConfig
from we_cfg.fdc.data_utils import create_human_pose_from_kpts


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


def save_debug_result(
    stereo_data: StereoData,
    outputs: List[dict],
    faces: np.ndarray,
    visualizer,
    output_dir: str,
    image_name: str,
) -> dict:
    """
    Save results organized by type in separate folders.
    
    Args:
        stereo_data: Stereo data
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
        'keypoints': os.path.join(output_dir, 'keypoints'),
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    saved_counts = {
        'meshes': 0,
        'overlays': 0,
        'bboxes': 0,
        'skeletons': 0,
        'keypoints': 0,
    }
    
    # Visualize 2D results with skeleton
    skeleton_images = visualize_2d_results(stereo_data.left_image, outputs, visualizer)
    
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
                stereo_data.left_image.copy(),
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
        img_bbox = stereo_data.left_image.copy()
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
    stereo_loader: StereoDataLoader,
    output_dir: str,
    hf_repo_id: str = "facebook/sam-3d-body-dinov3",
    detector_name: str = "vitdet",
    segmentor_name: str = "sam2",
    fov_name: str = "moge2",
    device: str = "cuda",
    detector_path: str = "",
    segmentor_path: str = "",
    mhr_path: str = "",
    ckpt_path: str = "",
    fov_path: str = "",
    create_videos: bool = True,
    video_fps: float = 30.0,
    frame_step: int = 10,
    ego_valid_dist: float = 0.5,
    debug_vis: bool = False,
):
    """
    Process a list of images with SAM 3D Body and save results.
    
    Args:
        video_loader: StereoDataLoader instance
        output_dir: Directory to save output files
        hf_repo_id: HuggingFace repository ID for the model
        detector_name: Name of detector to use
        segmentor_name: Name of segmentor to use
        fov_name: Name of FOV estimator to use
        device: Device to use (cuda/cpu)
        detector_path: Path to detector model (optional)
        segmentor_path: Path to segmentor model (optional)
        fov_path: Path to FOV estimator model (optional)
        debug_vis: Whether to save debug visualizations
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
        mhr_path=mhr_path,
        ckpt_path=ckpt_path,
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
    
    # Buffer
    idxs = []
    timestamps = []
    kpts_list = []
    human_poses_list = []
    kpts_scores_list = []

    for frame_idx in tqdm(stereo_loader.get_frame_list(frame_step=frame_step), desc="Processing frames"):
        try:
            # Load frame
            stereo_data: StereoData = stereo_loader[frame_idx]
            
            # Process the image with SAM 3D Body
            use_load_int = True
            if use_load_int:
                cam_int_tensor = torch.from_numpy(np.array(stereo_data.intrinsic)).to(device)[None, :, :]
            else:
                cam_int_tensor = None

            # Convert BGR to RGB
            # left_image_rgb = cv2.cvtColor(stereo_data.left_image, cv2.COLOR_BGR2RGB)
            left_image_rgb = stereo_data.left_image
            outputs = estimator.process_one_image(img=left_image_rgb, cam_int=cam_int_tensor)
            
            # Update buffer
            human_kpts = []
            human_poses = []
            human_kpts_dist = []
            for pid, person_output in enumerate(outputs):
                person_kpts = person_output["pred_keypoints_3d"] + person_output["pred_cam_t"][None, :]
                person_vertices = copy.deepcopy(person_output["pred_vertices"]) + person_output["pred_cam_t"][None, :]

                # # Rotate along x-axis by 180 degrees
                # rot_x_axis = R.from_euler('x', 180, degrees=True).as_matrix()
                # person_vertices = (rot_x_axis @ person_vertices.T).T

                person_kpts_dist = np.linalg.norm(person_kpts[0, ...], axis=-1)
                if person_kpts_dist > ego_valid_dist:
                    continue
                
                # Generate human pose object
                person_kpts = Kpt3D(
                    kpts=person_kpts,
                    kpts_scores=np.ones(len(person_kpts)),
                    kpts_names=[],
                    frame_idx=frame_idx,
                    timestamp=stereo_data.timestamp,
                    frame_name=stereo_data.frame_name,
                )
                human_pose_sam3d = create_human_pose_from_kpts(person_kpts, person_vertices, "sam3d")
                human_poses.append(human_pose_sam3d)
                human_kpts.append(person_kpts)
                human_kpts_dist.append(person_kpts_dist)
            # Select the closest human keypoints to the camera
            if len(human_kpts) > 0:
                closest_human_kpts = human_kpts[np.argmin(human_kpts_dist)]
                closest_human_pose = human_poses[np.argmin(human_kpts_dist)]
                ego_kpts = closest_human_kpts
                ego_kpts_scores = np.array([1.0] * closest_human_kpts.kpts.shape[0])
                ego_pose = closest_human_pose
            else:
                ego_kpts = np.zeros((70, 3))
                ego_kpts_scores = np.zeros((70,))
                ego_pose = HumanPose.get_invalid_human_pose()
            # Convert to Kpt3D
            idxs.append(frame_idx)
            timestamps.append(stereo_data.timestamp)
            kpts_list.append(ego_kpts)
            human_poses_list.append(ego_pose)
            kpts_scores_list.append(ego_kpts_scores)
            successful += 1

            if debug_vis:
                save_debug_result(
                    stereo_data,
                    outputs,
                    estimator.faces,
                    visualizer,
                    output_dir,
                    f"frame_{frame_idx:06d}",
                )
            
        except Exception as e:
            print(f"\nError processing {frame_idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1

    # Save human poses
    np.savez(os.path.join(output_dir, "output.npz"), humanoid_poses=human_poses_list)


def main():
    parser = argparse.ArgumentParser(description="Process images with SAM 3D Body")
    
    parser.add_argument(
        "--data_config", "-c",
        type=str,
        default="configs/config_0000_down.yaml",
        help="Path to data config file",
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="output",
        help="Output directory",
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
        "--data_root", "-d",
        type=str,
        default="/home/haonan/Projects/we_human_pose/data",
        help="Data root directory",
    )
    parser.add_argument(
        "--local_output_dir",
        action="store_true",
        help="Use local storage for output directory (default: output directory is in the data root)",
    )
    parser.add_argument(
        "--video_fps",
        type=float,
        default=30.0,
        help="Frames per second for output videos (default: 30.0)",
    )
    parser.add_argument(
        "--debug_vis",
        action="store_true",
        help="Save debug visualizations (default: debug visualizations are not saved)",
    )
    args = parser.parse_args()
    
    # Create data config & video loader
    with open(args.data_config, "r") as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
        data_config["data"]["data_root"] = args.data_root
        data_config = StereoDataConfig(**data_config["data"])
    video_loader = load_stereo_data_loader_from_cfg(data_config.rectify_config)
    output_dir = data_config.data_cache_dir + "/sam3d"

    human_recon_config: HumanReconConfig = HumanReconConfig.from_yaml(yaml_path=args.data_config)
    os.makedirs(output_dir, exist_ok=True)
    # Process images
    process_images(
        stereo_loader=video_loader,
        output_dir=output_dir,
        hf_repo_id=human_recon_config.hf_repo_id,
        detector_name=human_recon_config.detector_name,
        segmentor_name=human_recon_config.segmentor_name,
        fov_name=human_recon_config.fov_name,
        device=args.device,
        detector_path=human_recon_config.detector_path,
        segmentor_path=human_recon_config.segmentor_path,
        fov_path=human_recon_config.fov_path,
        create_videos=not args.no_videos,
        video_fps=args.video_fps,
        mhr_path=human_recon_config.mhr_path,
        ckpt_path=human_recon_config.ckpt_path,
        frame_step=human_recon_config.frame_step,
        ego_valid_dist=human_recon_config.ego_valid_dist,
        debug_vis=args.debug_vis,
    )


if __name__ == "__main__":
    main()

