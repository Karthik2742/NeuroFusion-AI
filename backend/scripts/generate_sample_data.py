"""
NeuroFusion AI - Sample Synthetic Multi-Class Dataset Generator
Generates realistic 3D synthetic Brain MRI NIfTI volumes organized into:
dataset/
└── brain/
    ├── normal/
    ├── tumor/
    └── stroke/
"""

import numpy as np
import nibabel as nib
from pathlib import Path

def generate_brain_mri(shape=(128, 128, 64), pathology="normal"):
    """Generate 3D Brain MRI volume with optional pathology (tumor, stroke)."""
    dx, dy, dz = shape
    x, y, z = np.ogrid[:dx, :dy, :dz]
    cx, cy, cz = dx / 2, dy / 2, dz / 2
    
    # Brain skull/tissue mask
    dist_sq = ((x - cx) / (cx * 0.85))**2 + ((y - cy) / (cy * 0.85))**2 + ((z - cz) / (cz * 0.85))**2
    brain_mask = dist_sq <= 1.0
    
    # Brain tissue intensity (Gray/White matter)
    tissue = np.sin(x / 6.0) * np.cos(y / 6.0) + np.sin(z / 4.0)
    data = (tissue - tissue.min()) / (tissue.max() - tissue.min()) * 500.0 + 100.0
    data = data * brain_mask
    
    mask = np.zeros(shape, dtype=np.float32)
    
    if pathology == "tumor":
        # Hyperintense tumor mass with necrosis core
        tx, ty, tz = int(cx + 15), int(cy + 15), int(cz)
        tumor_dist = ((x - tx)/12.0)**2 + ((y - ty)/12.0)**2 + ((z - tz)/10.0)**2
        tumor_mask = tumor_dist <= 1.0
        data[tumor_mask] += 450.0
        mask[tumor_mask] = 4.0  # Enhancing Tumor
        
    elif pathology == "stroke":
        # Ischemic lesion (hypointense wedge / lacunar infarct)
        sx, sy, sz = int(cx - 20), int(cy + 10), int(cz)
        stroke_dist = ((x - sx)/16.0)**2 + ((y - sy)/10.0)**2 + ((z - sz)/12.0)**2
        stroke_mask = stroke_dist <= 1.0
        data[stroke_mask] *= 0.3
        mask[stroke_mask] = 1.0  # Infarct core
        
    # Gaussian noise
    data += np.abs(np.random.normal(0, 12.0, shape)) * brain_mask
    return data.astype(np.float32), mask

def create_sample_dataset():
    root = Path("./dataset")
    affine = np.eye(4)
    affine[0, 0] = 1.0
    affine[1, 1] = 1.0
    affine[2, 2] = 1.5
    
    categories = {
        "brain": {
            "normal": lambda: generate_brain_mri(pathology="normal"),
            "tumor": lambda: generate_brain_mri(pathology="tumor"),
            "stroke": lambda: generate_brain_mri(pathology="stroke"),
        }
    }
    
    print("============================================================")
    print("      NeuroFusion AI - Multi-Class Dataset Generator        ")
    print("============================================================")
    
    for category, subcats in categories.items():
        for subcat, generator_fn in subcats.items():
            sub_dir = root / category / subcat
            sub_dir.mkdir(parents=True, exist_ok=True)
            
            volume, mask = generator_fn()
            
            # Save NIfTI volume
            vol_path = sub_dir / f"{category}_{subcat}_001_t2.nii.gz"
            nib.save(nib.Nifti1Image(volume, affine), str(vol_path))
            print(f"Saved: {vol_path}")
            
            # Save segmentation mask if pathology present
            if np.any(mask > 0):
                mask_path = sub_dir / f"{category}_{subcat}_001_seg.nii.gz"
                nib.save(nib.Nifti1Image(mask, affine), str(mask_path))
                print(f"Saved: {mask_path}")
                
    print("============================================================")
    print(" [SUCCESS] Multi-Class Dataset Structure Successfully Created!")
    print("============================================================")

if __name__ == "__main__":
    create_sample_dataset()
