# UGOD: Uncertainty-Guided Differentiable Opacity and Soft Dropout for Enhanced Sparse-View 3DGS

UGOD is an uncertainty-aware learning module designed to improve 3DGS under sparse-view conditions. It introduces a differentiable MLP that predicts per-Gaussian uncertainty and uses it to guide opacity modulation and soft dropout during training process to better generalisation and avoid overfitting.



## 🚀 Getting Started

### Training (with default UGOD setup)
## Environment Setup
```bash
You need to compile submodels first using:

   conda env create --file environment.yml
 
   conda activate gaussian_splatting
```
## Training Command
```bash
!python train.py -s /data_path --eval --iterations=6000
!python render.py -m /output_path
!python eval.py -m /output_path
```