import os
import sys
import cv2
import matplotlib.pyplot as plt

#==Team 12 pdf 파일의 Appendix-(b)에서 token 값 들고 와주시기 바랍니다==
os.environ["HF_TOKEN"] = "token"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
#=====================================================================

###경로 확인 부탁드립니다###
PROJECT_ROOT="C:/Users/Downloads/CV/sam-3d-body/notebook"
###########################

from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])

parent_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, parent_dir)

from utils import (
    setup_sam_3d_body, setup_visualizer,
    visualize_2d_results, visualize_3d_mesh, save_mesh_results,
    display_results_grid, process_image_with_mask
)

# Set up SAM 3D Body estimator
estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3")
# Set up visualizer
visualizer = setup_visualizer()

image_path = f"{PROJECT_ROOT}/original4.jpg"  # Relative to notebook folder
img_cv2 = cv2.imread(image_path)
outputs = estimator.process_one_image(image_path)

if outputs:
    # Get image name without extension
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # Create output directory
    output_dir = f"output/{image_name}"

    # Save all results (PLY meshes, overlay images, bbox images)
    ply_files = save_mesh_results(img_cv2, outputs, estimator.faces, output_dir, image_name)

    print(f"\n=== Saved Results for {image_name} ===")
    print(f"Output directory: {output_dir}")
    print(f"Number of PLY files created: {len(ply_files)}")

else:
    print("No results to save - no people detected")