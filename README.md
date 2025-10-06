# Vision Mamba for Permeability Prediction of Porous Media

**Software by:** Ali Kashefi (kashefi@stanford.edu) 

**Citation** <br>
If you use the code, please cite the following article: <br>

**[Physics-informed KAN PointNet: Deep learning for simultaneous solutions to inverse problems in incompressible flow on numerous irregular geometries](https://arxiv.org/abs/2504.06327)**

    @article{kashefi2025PhysicsInformedKANpointnet,
      title={Physics-informed KAN PointNet: Deep learning for simultaneous solutions to inverse problems in incompressible flow on numerous irregular 
      geometries},
      author={Kashefi, Ali and Mukerji, Tapan},
      journal={arXiv preprint arXiv:2504.06327},
      year={2025}}

**Abstract** <be>

Vision Mamba has recently received attention as an alternative to Vision Transformers (ViTs) for image classification. The network size of Vision Mamba scales linearly with input image resolution, whereas ViTs scale quadratically, a feature that improves computational and memory efficiency. Moreover, Vision Mamba requires a significantly smaller number of trainable parameters than traditional convolutional neural networks (CNNs), and thus, they can be more memory efficient. Because of these features, we introduce, for the first time, a neural network that uses Vision Mamba as its backbone for predicting the permeability of three-dimensional porous media. We compare the performance of Vision Mamba with ViT and CNN models across multiple aspects of permeability prediction and perform an ablation study to assess the effects of its components on accuracy. We demonstrate in practice the aforementioned advantages of Vision Mamba over ViTs and CNNs in the permeability prediction of three-dimensional porous media. We make the source code publicly available to facilitate reproducibility and to enable other researchers to build on and extend this work. We believe the proposed framework has the potential to be integrated into large vision models in which Vision Mamba is used instead of ViTs.

**Installation** <be>
This guide will help you set up the environment required to run the code. Follow the steps below to install the necessary dependencies.

**Step 1: Download and Install Miniconda**

1. Visit the [Miniconda installation page](https://docs.conda.io/en/latest/miniconda.html) and download the installer that matches your operating system.
2. Follow the instructions to install Miniconda.

**Step 2: Create a Conda Environment**

After installing Miniconda, create a new environment:

```bash
conda create --name myenv python=3.8
```

Activate the environment:

```bash
conda activate myenv
```

**Step 3: Install PyTorch**

Install PyTorch with CUDA 11.8 support:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Step 4: Install Additional Dependencies**

Install the required Python libraries:

```bash
pip3 install numpy matplotlib torchsummary
```

**Step 5: Download the Data** <be>

Use the following batch command to download the dataset (a NumPy array, approximately 42MB in size).

```bash
wget https://web.stanford.edu/~kashefi/data/HeatTransferData.npy
```

**Questions?** <br>
If you have any questions or need assistance, please do not hesitate to contact Ali Kashefi (kashefi@stanford.edu) via email. 

**About the Author** <br>
Please see the author's website: [Ali Kashefi](https://web.stanford.edu/~kashefi/) 

