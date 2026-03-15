
# Medical Image Classification (PathMNIST + EfficientNet-B0)

This project trains a medical image classification model using PyTorch on Google Colab GPU.  
The model is trained on the PathMNIST dataset from MedMNIST using a pretrained EfficientNet-B0 backbone.

---

## Environment

Platform: Google Colab  
GPU: Colab Free GPU (Tesla T4)  
Framework: PyTorch  

Libraries used:

- timm
- medmnist
- torchvision
- matplotlib
- tqdm

Install dependencies:

```bash
pip install medmnist timm
```

---

## Project Structure

```
medmnist_efficientnet_project/
├── src/
│   ├── data.py             # Handles data loading, transformations, and dataloaders
│   ├── model.py            # Defines the EfficientNet-B0 model and related components
│   └── train.py            # Contains training, validation, testing, and prediction logic
├── best_model.pth          # Saved best model weights during training
└── README.md               # Project overview and documentation
└── main_notebook.ipynb
```
---

## DatasetS

Dataset: PathMNIST (MedMNIST)S

| Split | Samples |
|------|--------|
| Train | 89,996 |
| Validation | 10,004 |
| Test | 7,180 |

Number of classes: 9  
Image size: 224 x 224

---

## Model

Backbone: EfficientNet-B0 (pretrained)

The final classification layer is replaced to output 9 classes.

---

## Training Configuration

| Parameter | Value |
|----------|------|
| Image Size | 224 |
| Batch Size | 32 |
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Weight Decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Epochs | 5 |
| Mixed Precision | Enabled |

---

## Data Augmentation

- RandomHorizontalFlip  
- RandomRotation  
- RandomResizedCrop  
- Normalization

---

## Training Logs

Epoch 1  
Train Loss: 0.1981  
Validation Accuracy: 0.9787  

Epoch 2  
Train Loss: 0.0872  
Validation Accuracy: 0.9844  

Epoch 3  
Train Loss: 0.0611  
Validation Accuracy: 0.9896  

Epoch 4  
Train Loss: 0.0398  
Validation Accuracy: 0.9909  

Epoch 5  
Train Loss: 0.0282  
Validation Accuracy: 0.9912  

Best Validation Accuracy: **0.991203518592563**

---

## Test Result

Test Accuracy: **0.9093314763231197**

---

## Saved Model

Model file: `best_model.pth`  
Saved path in Colab: `/content/best_model.pth`

---

## Quick run 

```bash
python run_model.py
```

## Notes

- Training performed on free Google Colab GPU.
- Mixed precision training used for faster training.
- EfficientNet-B0 used as pretrained backbone for transfer learning.
