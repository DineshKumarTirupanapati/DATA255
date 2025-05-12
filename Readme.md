# ResDiff: Fourier CNN and Diffusion UNet for Microscopy Image Super-Resolution

This repository contains the implementation of ResDiff, a hybrid model combining Fourier-based CNN and Diffusion UNet for super-resolution of biological microscopy images.

## Repository Structure

- `aplit_dataset.ipynb`: Splits the dataset into train, validation, and test sets using an 80/10/10 ratio.
- `dataset.py`: Creates high-resolution (256×256) and low-resolution (64×64) image pairs for training.
- `resdiff.py`: Defines the core architecture of the ResDiff model, integrating FourierCNN and DiffusionUNet.
- `losses.py`: Implements the full composite loss function (MSE, SSIM, Fourier, Phase, VGG) used during training.
- `metrics.py`: Contains evaluation metrics including PSNR, SSIM, LPIPS, Fourier Cosine Similarity, and Phase Consistency.
- `train_fresh.ipynb`: Main training script for training the ResDiff model from scratch.
- `eval.ipynb`: Notebook to compute and visualize final evaluation results on the test dataset.

## Datasets

- `datasets/`: Preprocessed dataset folder (train, val, test splits)  
  [View on GitHub](https://github.com/DineshKumarTirupanapati/DATA255/tree/main/datasets)

## Results

- `results/`: Contains model checkpoints and generated sample outputs  
  - `checkpoints/`: Stores ResDiff model weights per epoch  
  - `samples/`: Contains qualitative output images  
  [View on GitHub](https://github.com/DineshKumarTirupanapati/DATA255/tree/main/results)

## Evaluation Results

- `Evaluation_results_fresh_best/`:  
  - `evaluation_results.txt`: Final evaluation metric scores  
  - Other files: Evaluation image comparisons  
  [View on GitHub](https://github.com/DineshKumarTirupanapati/DATA255/tree/main/evaluation_results_fresh_best)

## Model Highlights

- Combines frequency-domain and spatial-domain learning
- Efficient and lightweight – trainable on consumer GPUs
- Achieves better perceptual quality (LPIPS) and structural integrity compared to U-Net

---

**Authors**: Dinesh Kumar Tirupanapati, Ava Xia, Bo Zhi  
**Affiliation**: San Jose State University

For questions or collaboration, feel free to open an issue or reach out.
