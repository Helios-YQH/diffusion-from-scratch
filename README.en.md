# DDPM Diffusion Model Project

## Project Introduction

This project implements an image generation system based on the Denoising Diffusion Probabilistic Model (DDPM). DDPM is one of the most关注的 (popular/focused) generative models in recent years, capable of generating high-quality image samples by simulating the gradual diffusion and denoising processes of data.

The project adopts a modular design. Core components include the UNet neural network architecture, diffusion scheduler, training pipeline, and visualization tools. It supports training and sampling on datasets such as MNIST and CelebA.

## Core Features

**Diffusion Model Core**: Implements the complete DDPM algorithm, including the forward diffusion process and the reverse denoising sampling process. The forward process gradually adds Gaussian noise to the data, while the reverse process learns denoising operations through a neural network.

**UNet Architecture**: Employs a U-shaped network with time-conditioned embedding as the denoising model. It includes residual blocks, attention modules, and upsampling/downsampling modules. Residual blocks are used for feature extraction, attention modules capture long-range dependencies, and the U-shaped structure ensures multi-scale information fusion.

**Time Condition Embedding**: Uses Sinusoidal Time Embedding to encode diffusion steps into high-dimensional vectors. These are injected into various layers of the network via an MLP, enabling the model to adaptively adjust according to different denoising stages.

**Multi-Dataset Support**: Includes processing pipelines for the built-in MNIST handwritten digit dataset and the CelebA face dataset, which can be flexibly switched via configuration files. MNIST is suitable for rapid verification and small-scale experiments, while CelebA is used for generating more complex natural images.

**Training Tools**: Provides a complete training framework, including learning rate scheduling, model checkpoint saving/loading, early stopping mechanisms, and sample grid generation. Supports resuming training from a checkpoint.

## Environment Requirements

The project is developed based on Python 3.8+. Core dependencies include PyTorch version 2.0 or higher, TorchVision for data processing, and PyYAML for configuration file parsing. Using an NVIDIA GPU is recommended for training on the CelebA dataset; a VRAM of at least 8GB is suggested for optimal results.

## Installation Steps

After cloning the repository to your local machine, create and activate a virtual environment. It is recommended to use conda or venv to manage dependencies. Install project dependencies via pip. If using GPU training, ensure the CUDA environment is configured correctly.

```bash
git clone <repository_url>
cd 26summer
pip install -r requirements.txt
```

## Quick Start

The project provides preset configuration files for training on different datasets. Use `main.py` as the entry point program and specify the configuration file path via the `--config` parameter.

```bash
# Train using MNIST configuration
python main.py --config configs/mnist.yml

# Train using CelebA configuration
python main.py --config configs/celeba.yml
```

During training, sample grids will be generated periodically to the specified directory, and checkpoint files are saved in the folder corresponding to the run name. After training is completed, you can use the functions in `demo.py` for sampling and visualization.

## Configuration File Description

Configuration files are in YAML format and contain settings for model architecture parameters, training hyperparameters, dataset paths, etc. The MNIST configuration is suitable for rapid experiments, defaulting to smaller batch sizes and shorter training epochs. The CelebA configuration is optimized for high-quality face generation, utilizing larger batch sizes and longer training durations.

Key configuration items include: the model's base channel count, number of diffusion steps, learning rate, batch size, image dimensions, dataset root directory path, etc. Modifying these parameters allows adaptation to different hardware conditions and experimental requirements.

## Core Module Details

**diffusion.py** implements the core logic of DDPM. The `forward_diffusion` method accepts raw images and time steps, adding corresponding levels of Gaussian noise at the specified step. The `training_loss` method calculates the denoising loss function, using Mean Squared Error to measure the difference between predicted noise and actual added noise. The `sample` method executes the reverse sampling process, gradually denoising from pure noise to generate new images.

**model.py** defines the complete UNet denoising model. `SinusoidalTimeEmbedding` generates embedding vectors for time steps using sine and cosine functions; this encoding method is borrowed from Transformer architectures. `TimeMLP` maps time embeddings to dimensions suitable for network operations. `ResidualBlock` combines time condition information and image features; residual connections aid in training deep networks. `AttentionBlock` employs multi-head self-attention mechanisms to enhance the model's modeling capability for image details. `DownSample` and `UpSample` implement compression and restoration of feature maps, working with skip connections to retain multi-scale information.

**train.py** contains the complete implementation of the training pipeline. The `extract_celeb` and `preprocess_celeb` functions handle raw CelebA image data. The `load_mnist` function loads and normalizes the MNIST dataset. `save_sample_grid` stitches generated samples into a grid image for intuitive comparison. Checkpoint-related functions support resuming after training interruption and saving the best model.

**demo.py** provides convenient visualization tools. `to_image` and `show_image` are used to convert tensors into visualizable image formats. `capture_reverse` visualizes intermediate results of the reverse denoising process, intuitively showing the image gradually becoming clear from noise. `demo_animate` generates animation demonstrations of the denoising process.

## Model Checkpoints

During training, model states are saved in the `checkpoints` directory. It is recommended to use the latest checkpoint (`latest`) to resume interrupted training, and the best checkpoint (`best`) for final sampling. Checkpoints contain model parameters, optimizer states, learning rate scheduler states, current epoch, loss values, and other information.

## Algorithm Principles

The core idea of diffusion models is to learn the inverse transformation of the data distribution. The forward diffusion process gradually adds noise to the data over T time steps, where the noise level at each step is controlled by a pre-defined scheduler. The reverse denoising process starts from a pure noise distribution, where the neural network predicts the noise to be removed at each step, gradually restoring samples of the data distribution.

The training objective is to make the noise predicted by the neural network as close as possible to the actual added noise, using a Mean Squared Loss function. Derived via the Variational Lower Bound, this objective is equivalent to maximizing the Evidence Lower Bound (ELBO). During sampling, iterative updates are performed using a formula where the mean is the predicted value plus the current state, and the variance is determined by the scheduler.

## Project Structure

```
├── configs/           # Configuration file directory
│   ├── celeba.yml    # CelebA dataset configuration
│   └── mnist.yml     # MNIST dataset configuration
├── diffusion.py      # DDPM diffusion process implementation
├── model.py          # UNet denoising model definition
├── train.py          # Training pipeline and utility functions
├── demo.py           # Visualization and demonstration code
├── main.py           # Program entry point
└── test_check.py     # Test script
```

## License

This project follows an open-source license. For specific license information, please refer to the LICENSE file.