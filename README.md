# GANs Implementation - Photo to Sketch Conversion

A PyTorch implementation of Generative Adversarial Networks (GANs) for converting facial photos to sketches using the CUHK Face Sketch Database.

## Overview

This project implements a GAN-based approach to transform realistic facial photographs into artistic sketches. The model learns the mapping between photo and sketch domains through adversarial training.

## Features

- **Custom Dataset Loader**: Handles paired photo-sketch data from CUHK Face Sketch Database
- **Generator Network**: Converts input photos to sketch-style images
- **Discriminator Network**: Distinguishes between real sketches and generated ones
- **GPU Acceleration**: CUDA support for faster training
- **Data Augmentation**: Includes image preprocessing and normalization
- **Visualization Tools**: Built-in plotting for training progress monitoring

## Dataset

- **Source**: CUHK Face Sketch Database (CUFS)
- **Size**: 188 paired photo-sketch samples
- **Format**: RGB images resized to 128x128 pixels
- **Preprocessing**: Normalized to [-1, 1] range for stable training

## Architecture

### Generator Network
```python
Input: RGB Photo (3, 128, 128)
├── Conv2d (3→64) + ReLU
├── Conv2d (64→128) + ReLU  
├── ConvTranspose2d (128→64) + ReLU
└── ConvTranspose2d (64→3) + Tanh
Output: Generated Sketch (3, 128, 128)
```

### Key Components
- **Encoder-Decoder Architecture**: Compresses photo features then reconstructs as sketch
- **Skip Connections**: Preserves fine details during generation
- **Adversarial Loss**: Ensures realistic sketch generation
- **L1 Loss**: Maintains pixel-level similarity to target sketches

## Requirements

```python
torch>=1.7.0
torchvision>=0.8.0
PIL>=8.0.0
matplotlib>=3.3.0
numpy>=1.19.0
```

## Installation & Setup

1. **Clone or download** the notebook
2. **Install dependencies**:
   ```bash
   pip install torch torchvision pillow matplotlib numpy
   ```
3. **Download CUHK Face Sketch Database**:
   - Photos directory: `/kaggle/input/cuhk-face-sketch-database-cufs/photos`
   - Sketches directory: `/kaggle/input/cuhk-face-sketch-database-cufs/sketches`
4. **Update dataset paths** in the notebook if needed

## Usage

### Running the Notebook
1. Open `gans-implementations.ipynb` in Jupyter or Google Colab
2. Execute cells sequentially to:
   - Load and preprocess the dataset
   - Define generator and discriminator models
   - Train the GAN network
   - Generate sketch outputs

### Key Functions

```python
# Custom dataset for paired photo-sketch data
dataset = SketchDataset(photo_dir, sketch_dir, transform)

# Generator model for photo-to-sketch conversion
generator = Generator()

# Training loop with adversarial loss
for epoch in range(num_epochs):
    # Train discriminator and generator
    pass
```

### Customization Options

- **Modify Architecture**: Adjust layer sizes and depths
- **Change Loss Functions**: Experiment with different adversarial losses
- **Hyperparameter Tuning**: Adjust learning rates, batch sizes
- **Data Augmentation**: Add rotation, flipping, color jittering

## Training Process

1. **Data Loading**: Paired photos and sketches loaded with transforms
2. **Generator Training**: Learns photo→sketch mapping
3. **Discriminator Training**: Learns to distinguish real vs. fake sketches
4. **Adversarial Process**: Both networks improve through competition
5. **Evaluation**: Generated sketches compared to ground truth

## Results & Evaluation

- **Training Metrics**: Generator and discriminator losses
- **Visual Quality**: Generated sketches vs. real sketches
- **Feature Preservation**: Facial structure and detail retention
- **Style Transfer**: Artistic sketch-like appearance

## Technical Implementation

### Performance Optimizations
- **GPU Acceleration**: Automatic CUDA detection and usage
- **Efficient Data Loading**: Optimized DataLoader with batching
- **Memory Management**: Proper tensor handling and cleanup

### Model Architecture Details
- **Convolutional Layers**: Feature extraction and reconstruction
- **Batch Normalization**: Training stability and convergence
- **Activation Functions**: ReLU for hidden layers, Tanh for output
- **Skip Connections**: Preserve high-frequency details

## Applications

- **Artistic Style Transfer**: Convert photos to artistic sketches
- **Law Enforcement**: Generate sketches from facial photos
- **Entertainment**: Create artistic portraits from photos
- **Education**: Demonstrate GAN concepts and image translation
- **Research**: Foundation for advanced image-to-image translation

## Future Enhancements

- [ ] Implement Progressive Growing GANs for higher resolution
- [ ] Add CycleGAN for unpaired training
- [ ] Incorporate attention mechanisms
- [ ] Experiment with different loss functions (WGAN, LSGAN)
- [ ] Add style control parameters
- [ ] Implement real-time inference
- [ ] Create web interface for easy usage

## Troubleshooting

### Common Issues
- **CUDA out of memory**: Reduce batch size or image resolution
- **Dataset path errors**: Verify correct directory structure
- **Slow training**: Ensure GPU is being utilized
- **Poor quality outputs**: Adjust learning rates or model architecture

### Performance Tips
- Use GPU for faster training (10-50x speedup)
- Monitor training curves to detect convergence
- Experiment with different hyperparameters
- Save model checkpoints regularly

## Contributing

Contributions are welcome! Areas for improvement:
- Model architecture enhancements
- Training stability improvements
- Evaluation metrics implementation
- Documentation and examples

## References

- **CUHK Face Sketch Database**: Original dataset source
- **Pix2Pix**: Image-to-image translation inspiration
- **GANs Paper**: Goodfellow et al., 2014
- **Deep Learning**: PyTorch documentation and tutorials

## License

This project is available under the MIT License. Please cite the CUHK Face Sketch Database if used for research.

---

*Built for exploring the fascinating intersection of computer vision and artistic style transfer.* 
