# dataset_setup.py
import os
import shutil
from glob import glob
from pathlib import Path

def prepare_cyclegan_layout(kaggle_extract_dir, out_dir):
    """
    Attempts to find photos and sketches in the kaggle dataset directory
    and create a CycleGAN folder layout:
      out_dir/
        trainA/   # sketches
        trainB/   # photos
        testA/
        testB/
    Edit heuristics if your dataset layout differs.
    """
    os.makedirs(out_dir, exist_ok=True)
    trainA = Path(out_dir) / 'trainA'
    trainB = Path(out_dir) / 'trainB'
    testA = Path(out_dir) / 'testA'
    testB = Path(out_dir) / 'testB'
    for d in [trainA, trainB, testA, testB]:
        d.mkdir(parents=True, exist_ok=True)

    # Heuristics: filenames containing 'sketch' go to A; others to B.
    files = glob(os.path.join(kaggle_extract_dir, '**', '*.*'), recursive=True)
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    countA = countB = 0
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext not in img_exts:
            continue
        name = os.path.basename(f).lower()
        if 'sketch' in name or 'sketches' in name or name.endswith('_s.png') or 'pencil' in name:
            shutil.copy(f, trainA / os.path.basename(f))
            countA += 1
        else:
            shutil.copy(f, trainB / os.path.basename(f))
            countB += 1

    print(f"Copied {countA} files to {trainA}")
    print(f"Copied {countB} files to {trainB}")
    print("If numbers look wrong, manually inspect kaggle_extract_dir and adjust heuristics.")

if __name__ == "__main__":
    # Example usage when run in Colab:
    # prepare_cyclegan_layout('/content/person-face-sketches', '/content/dataset')
    import sys
    if len(sys.argv) < 3:
        print("Usage: python dataset_setup.py <kaggle_extract_dir> <out_dir>")
    else:
        prepare_cyclegan_layout(sys.argv[1], sys.argv[2])
