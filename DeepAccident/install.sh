#!/bin/bash


echo "=================================="
echo "Step 1: remove old environment (if any)"
echo "=================================="
conda env remove -n deepaccident -y

echo ""
echo "=================================="
echo "Step 2: create conda environment"
echo "=================================="
conda create -n deepaccident python=3.7 -y

echo ""
echo "=================================="
echo "Step 3: activate environment"
echo "=================================="
conda activate deepaccident

echo ""
echo "=================================="
echo "Step 4: install PyTorch 1.10.2 + CUDA 11.3"
echo "=================================="
pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

echo ""
echo "=================================="
echo "Step 5: verify PyTorch and CUDA"
echo "=================================="
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "=================================="
echo "Step 6: install mmcv-full 1.3.16 (prebuilt)"
echo "=================================="
pip install mmcv-full==1.3.16 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

echo ""
echo "=================================="
echo "Step 7: install mmdet and mmsegmentation"
echo "=================================="
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1

echo ""
echo "=================================="
echo "Step 8: install other dependencies"
echo "=================================="
pip install scikit-image==0.18.3 "networkx>=2.2,<2.3" numba==0.48.0 setuptools==59.5.0
pip install nuscenes-devkit plyfile tensorboard
pip install "trimesh>=2.35.39,<2.35.40" torchmetrics==0.9.3 fvcore sympy

echo ""
echo "=================================="
echo "Step 9: verify installation"
echo "=================================="
python -c "
import torch
import mmcv
import mmdet
import mmseg
print('=' * 50)
print('Environment verified successfully.')
print('=' * 50)
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'MMCV: {mmcv.__version__}')
print(f'MMDetection: {mmdet.__version__}')
print(f'MMSegmentation: {mmseg.__version__}')
print('=' * 50)
"

echo ""
echo "=================================="
echo "Installation complete."
echo "=================================="
echo "Environment name: deepaccident"
echo "Python: 3.7"
echo "PyTorch: 1.10.2+cu113"
echo "CUDA: 11.3"
echo "MMCV: 1.3.16"
echo ""
echo "Activate with:"
echo "conda activate deepaccident"
