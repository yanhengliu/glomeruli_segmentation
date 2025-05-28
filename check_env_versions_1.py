#!/usr/bin/env python3
"""
check_versions.py

Inspect and print the versions of core libraries used in the maskrcnn_train scripts,
plus the Python interpreter version.
"""

import sys
import importlib

def print_version(label, module_name, version_attr="__version__"):
    """
    Attempt to import `module_name` and print `label: version`.
    If the module isn’t installed or has no attribute, report accordingly.
    """
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, version_attr, None)
        if version is None:
            # Some packages store their version elsewhere
            version = getattr(module, "VERSION", "Unknown")
        print(f"{label:15}: {version}")
    except ImportError:
        print(f"{label:15}: NOT INSTALLED")

def main():
    # Python version
    print(f"{'Python Runtime':15}: {sys.version.split()[0]}\n")

    # Core ML / vision libraries
    print_version("Torch",       "torch")
    print_version("Torchvision", "torchvision")
    print_version("NumPy",       "numpy")
    print_version("Pillow",      "PIL")
    print_version("scikit-image","skimage")
    print_version("tqdm",        "tqdm")
    print_version("TensorBoard", "tensorboard")

    # You can add more libraries here if needed:
    # e.g. print_version("rasterio", "rasterio")

if __name__ == "__main__":
    main()
