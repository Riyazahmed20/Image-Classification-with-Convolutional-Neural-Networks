# Image-Classification-with-Convolutional-Neural-Networks

# Image Classification using PyTorch

## Objective

To develop an end-to-end deep learning pipeline that classifies images into different categories using Convolutional Neural Networks (CNNs) in PyTorch. This project demonstrates how to load and preprocess image data, build and train a CNN, evaluate model performance, and visualize the results.

---

## Project Description

Image classification is a core task in computer vision with applications in medical diagnostics, object detection, and facial recognition. This notebook provides a hands-on walkthrough of building an image classifier from scratch using PyTorch.

### 🔍 Key Features:
- Load image datasets using `torchvision.datasets.ImageFolder`
- Apply data augmentation and normalization with `transforms.Compose`
- Build a custom CNN using `nn.Sequential` or `nn.Module`
- Train and validate the model with mini-batches using `DataLoader`
- Plot training and validation loss curves
- Evaluate model performance on unseen test images
- Display predictions and compare with ground truth

You can customize this notebook easily for:
- Multi-class classification tasks
- Binary classification
- Real-world image datasets
- Transfer learning (e.g., ResNet, VGG)

---

## How to Run
- Clone the Repository: git clone https://github.com/yourusername/Image-Classification-PyTorch.git
cd Image-Classification-PyTorch

## Dependencies

Install all necessary packages before running the notebook:

```bash
pip install torch torchvision matplotlib numpy scikit-learn
```

## Execute Cells

1. Load and preprocess the data

2. Build and train the CNN model

3. Monitor performance through loss and accuracy

4. Predict and visualize test images

## Plotting the Training Loss Curve
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

