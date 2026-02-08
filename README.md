# Accelerating Algebraic Multigrid (AMG) with Deep Learning

**PACS Course Project** - Learning optimal AMG parameters using Graph Neural Networks

This project implements a unified deep learning framework for accelerating Algebraic Multigrid (AMG) solvers by predicting optimal coarsening parameters and interpolation operators (or prolongation matrices).

---

## Project Overview

Algebraic Multigrid (AMG) methods are powerful iterative solvers for large sparse linear systems. However, their performance heavily depends on algorithmic parameters:
- **Theta (θ)**: Strength threshold for coarse/fine splitting
- **Interpolation operators (P)**: Prolongation matrices for grid transfer

This project uses **Convolutional Neural Networks (CNNs)** and **Graph Neural Networks (GNNs)** to learn optimal AMG parameters from problem characteristics, achieving:
- ✅ **Automated parameter tuning** (no manual configuration)
- ✅ **Faster convergence** (better coarse grids)
- ✅ **Generalization across problem types** (diffusion, elasticity, Stokes)

---

## Architecture: Two-Stage Pipeline

### Stage 1: Theta Prediction (C/F Splitting)
- **Input**: Sparse matrix graph (edge features, node degrees)
- **Output**: Optimal theta value for Ruge-Stüben C/F splitting
- **Model**: CNN and GNN
- **Training**: The samples with varied parameters

### Stage 2: P-Value Prediction (Interpolation)
- **Input**: Matrix + C/F splitting + strength matrix + baseline P
- **Output**: Improved prolongation matrix
- **Model**: Message-Passing Neural Network (MPNN)
- **Training**: The samples with ground-truth AMG operators

### Unified Model
- Sequential pipeline: Stage 1 → Stage 2
- Deployable for production solvers

---

## Project Structure

```
PACSProject/
├── include/                        # C++ headers
│   ├── DiffusionModel.hpp            # Diffusion problem
│   ├── ElasticModel.hpp              # Elasticity problem
│   ├── StokesModel.hpp               # Stokes flow problem
│   ├── GraphLaplacianModel.hpp       # Graph Laplacian problem
│   ├── GraphLaplacianModelEigen.hpp  # Eigen implementation
│   ├── AMGOperators.hpp              # AMG algorithms (C/F, P, S)
│   ├── Pooling.hpp                   # CNN pooling operators
│   ├── NPYWriter.hpp                 # NumPy binary format writer
│   ├── BatchNPYWriter.hpp            # Batch numpy binary format writer
│   ├── NPZWriter.hpp                 # NumPy compressed format writer
│   └── UnifiedDataGenerator.hpp
│
├── src/                           # C++ data generation
│   └── generate_amg_data.cpp     # Main source file to generata data

│
├── model/                         # Python neural network models
│   ├── cnn_model.py              # CNN for pooled matrix images
│   ├── gnn_model.py              # GNN for graph representations
│   ├── p_value_model.py          # MPNN for P-value prediction
│   └── unified_model.py          # Two-stage unified model
│
├── data/                          # Python data loaders
│   ├── cnn_data_processing.py    # CNN dataset loader
│   ├── gnn_data_processing.py    # GNN dataset loader (CSV)
│   └── unified_data_processing.py # Unified pipeline data loader
│
├── data_loader_npy.py             # NPY/NPZ data loaders
│
├── train_stage1.py                # Train theta prediction (Stage 1)
├── train_stage2.py                # Train P-value prediction (Stage 2)
├── train_unified.py               # Train full pipeline
├── evaluate.py                    # Model evaluation and metrics
│
├── datasets/                      # Generated datasets
│
└── weights/                       # Trained model weights
```

---

## Quick Start

### Prerequisites

- **C++ Compiler**: GCC 9+ or Clang 10+
- **CMake**: 3.15+
- **deal.II**: 9.5+ (with PETSc, MPI support)
- **PETSc**: 3.18+
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **PyTorch Geometric**: 2.3+

### Installation

#### 1. Install PETSc

```bash
# Method 1: Package manager (recommended)
sudo apt install petsc-dev

# Method 2: Build from source (see detailed instructions below)
```

#### 2. Install deal.II

```bash
# Download deal.II
cd ~/PACS
wget https://github.com/dealii/dealii/releases/download/v9.5.2/dealii-9.5.2.tar.gz
tar -xf dealii-9.5.2.tar.gz
cd dealii-9.5.2

# Build and install
mkdir build && cd build
cmake ../ \
    -DCMAKE_INSTALL_PREFIX=~/PACS/dealii-install \
    -DDEAL_II_WITH_PETSC=ON \
    -DDEAL_II_WITH_MPI=ON \
    -DDEAL_II_WITH_P4EST=ON \
    -DDEAL_II_WITH_TRILINOS=OFF
make -j$(nproc)
make install
```

#### 3. Build Project

```bash
cd ~/PACS/PACSProject
mkdir build && cd build
cmake ..
make -j$(nproc)
```

#### 4. Setup Python Environment

```bash
# Create virtual environment
python3 -m venv path/to/your/venv
source path/to/your/venv/bin/activate

# Install dependencies
pip install torch torchvision
pip install torch-geometric
pip install numpy pandas scipy matplotlib tqdm
```

---

## Dataset Generation

### Unified Generator

All datasets are generated using the unified `generate_amg_data` executable in the build folder:

```bash
build/generate_amg_data [OPTIONS]

Required Arguments:
  -p, --problem TYPE        Problem type: D|E|S|GL|SC
  -s, --split SPLIT         Dataset split: train|test
  -f, --format FORMAT       Output format: theta-cnn|theta-gnn|p-value|all
  -c, --scale SCALE         Dataset scale: small|medium|large|xlarge

Optional Arguments:
  -o, --output-dir DIR      Output directory (default: ./datasets/unified)
  -t, --threads NUM         OpenMP threads (default: auto)
  --seed SEED               Random seed (default: 42)
  -v, --verbose             Verbose progress output
  -h, --help                Show help message
  --csv                     Store data in CSV format (default: NPZ format)
```

### Output Formats

- **theta-cnn**: CNN-based theta prediction
- **theta-gnn**: GNN-based theta prediction
- **p-value**: MPNN-based P-value prediction
- **all**: Complete training pipeline

### Dataset Scales

#### FEM Problems (D, E, S)

| Scale | Diffussion | Elastic | Stokes | Use Case |
|-------|-----------|-----------|-----------|----------|
| **small** | 50 | 60 | 60 | Quick testing |
| **medium** | 450 | 540 | 1,080 | Validation |
| **large** | 1,200 | 2,500 | 3,000 | Full training |
| **xlarge** | 2,560 | 4,480 | 7,680 | Production |

#### Graph Problems (Graph Laplacian, Spectral Clustering)

| Scale | Samples | Nodes/Graph | Use Case |
|-------|---------|-------------|----------|
| **small** | 50 | 64 | Quick testing |
| **medium** | 500 | 128 | Validation |
| **large** | 2,000 | 256 | Full training |
| **xlarge** | 10,000 | 512 | Production |

### Examples

```bash
# Generate small diffusion training set
build/generate_amg_data -p D -s train -f all -c small --csv

# Generate xlarge graph Laplacian test set
build/generate_amg_data -p GL -s test -f theta-gnn -c xlarge --threads 8 --csv

# Generate medium elastic training with verbose output
build/generate_amg_data -p E -s train -f p-value -c medium -v --csv

# Generate all formats for Stokes, large scale
build/generate_amg_data -p S -s train -f all -c large --threads 8 --csv

# Spectral clustering with custom seed
build/generate_amg_data -p SC -s test -f all -c medium --seed 12345 --csv
```

### Output Structure

```
datasets/unified/
├── train/raw/
│   ├── theta_cnn/
│   │   ├── train_D.csv
│   │   ├── train_E.csv
│   │   ├── train_S.csv
│   │   ├── train_GL.csv
│   │   └── train_SC.csv
│   ├── theta_gnn/
│   │   └── (same structure)
│   └── p_value/
│       └── (same structure)
└── test/raw/
    └── (same structure as train)
```

---

## NPY/NPZ Binary Format

### Overview

We've implemented **high-performance NPY/NPZ binary format** for both data generation and training, achieving significant performance improvements:

- ✅ **Faster data generation**: For Graph Laplacian problem, 334 samples/s (NPZ) vs 64 samples/s (CSV)
- ✅ **Faster data loading**: Binary I/O eliminates CSV parsing overhead
- ✅ **Smaller file sizes**: Binary compression reduces storage
- ✅ **Complete pipeline support**: All problem types (D, E, S, GL, SC) and all stages

### Data Generation with NPZ

```bash
# Generate NPZ data (default, 5× faster)
./build/generate_amg_data -p GL -s train -f theta-gnn -c small -t 8
./build/generate_amg_data -p GL -s test -f theta-gnn -c small -t 8 

# Generate P-value data for Stage 2
./build/generate_amg_data -p GL -s train -f p-value -c medium -t 8
./build/generate_amg_data -p GL -s test -f p-value -c medium -t 8

# Generate all formats
./build/generate_amg_data -p D -s train -f all -c large -t 8

# CSV format (legacy, if needed)
./build/generate_amg_data -p GL -s train -f theta-gnn -c small --csv
```

### Training with NPY Format

**Stage 1 (Theta Prediction):**
```bash
python train_stage1.py \
    --dataset datasets/unified \
    --train-file train_GL \
    --test-file test_GL \
    --model GNN \
    --use-npy \
    --epochs 100 \
    --batch-size 64
```

**Full Two-Stage Pipeline:**
```bash
python train_unified.py \
    --dataset datasets/unified \
    --train-file train_GL \
    --test-file test_GL \
    --stage1-model GNN \
    --use-npy \
    --epochs-stage1 50 \
    --epochs-stage2 100
```

**Important:** When using `--use-npy`, specify problem types without `.csv` extension (e.g., `train_GL`, not `train_GL.csv`).

### NPZ File Structure

**Directory Layout:**
```
datasets/unified/
├── train/raw/
│   ├── theta_gnn_npy/          # NPZ format (recommended)
│   │   ├── train_D/            # Diffusion
│   │   │   ├── sample_00000.npz
│   │   │   ├── sample_00001.npz
│   │   │   └── ...
│   │   ├── train_E/            # Elastic
│   │   ├── train_S/            # Stokes
│   │   ├── train_GL/           # Graph Laplacian
│   │   └── train_SC/           # Spectral Clustering
│   ├── p_value_npy/            # NPZ format (recommended)
│   │   └── (same structure)
│   ├── theta_gnn/              # CSV format (legacy)
│   └── p_value/                # CSV format (legacy)
└── test/raw/
    └── (same structure)
```

**NPZ File Contents (theta_gnn):**
```python
{
    'edge_index': (2, num_edges),      # Sparse graph edges in COO format
    'edge_attr': (num_edges,),         # Edge values
    'theta': scalar,                    # Theta parameter
    'y': scalar,                        # rho (convergence factor)
    'metadata': [n, rho, h, epsilon]   # Problem metadata
}
```

**NPZ File Contents (p_value):**
```python
{
    'A_values', 'A_row_ptr', 'A_col_idx',    # System matrix (CSR)
    'coarse_nodes',                           # C/F splitting
    'P_values', 'P_row_ptr', 'P_col_idx',    # Prolongation matrix (CSR)
    'S_values', 'S_row_ptr', 'S_col_idx',    # Smoother matrix (CSR)
    'metadata': [n, theta, rho, h]            # Problem metadata
}
```

### Migration from CSV

**Backward Compatibility:**
- CSV loaders still work (omit `--use-npy` flag)
- Can mix CSV and NPY in different experiments
- NPY recommended for all new work

**Converting Workflow:**
1. Regenerate datasets with NPZ format (default)
2. Add `--use-npy` flag to training scripts
3. Update file paths to use problem types without `.csv`


---

## Training

### Stage 1: Theta Prediction

```bash
source venv/bin/activate

python train_stage1.py \
    --model GNN \
    --dataset datasets/unified \
    --epochs 100 \
    --batch-size 64 \
    --hidden-channels 128 \
    --lr 0.001 \
    --save-dir weights/xlarge/stage1 \
    --experiment-name gnn_xlarge_10k
```

**Expected results** (10,240 samples):
- Train MSE: ~0.001-0.005
- Test MSE: ~0.002-0.008
- Training time: 1-2 hours (100 epochs)

### Stage 2: P-Value Prediction

```bash
python train_stage2.py \
    --dataset datasets/unified \
    --epochs 150 \
    --batch-size 32 \
    --latent-size 128 \
    --num-message-passing 4 \
    --lr 0.003 \
    --save-dir weights/xlarge/stage2 \
    --experiment-name pvalue_xlarge_10k
```

**Expected results**:
- Convergence after ~100-150 epochs
- Training time: 2-3 hours

### Unified Pipeline (End-to-End)

```bash
python train_unified.py \
    --dataset=datasets/unified \
    --stage1-model=GNN \
    --epochs-stage1=100 \
    --epochs-stage2=150 \
    --batch-size-stage1=64 \
    --batch-size-stage2=32 \
    --hidden-channels=128 \
    --latent-size=128 \
    --save-dir=weights/xlarge/unified \
    --experiment-name=amg_xlarge_10k
```

**Total training time**: 3-5 hours

---

## Evaluation

```bash
# Evaluate Stage 1 model
python evaluate.py \
    --model-type stage1 \
    --model-path weights/xlarge/stage1/best_model.pt \
    --dataset datasets/unified \
    --hidden-channels 128

# Evaluate Stage 2 model
python evaluate.py \
    --model-type stage2 \
    --model-path weights/xlarge/stage2/best_model.pt \
    --dataset datasets/unified

# Evaluate unified model
python evaluate.py \
    --model-type unified \
    --model-path weights/xlarge/unified/unified_model_final.pt \
    --dataset datasets/unified
```
---

## Detailed Installation
```bash
# Step 1: Install PETSc
# First method: install it using package manager(Recommendation)
sudo apt install petsc-dev
# Second method: install it manually
# 1.1 Install required packages
sudo apt install -y build-essential cmake gfortran libopenmpi-dev openmpi-bin libblas-dev liblapack-dev libfftw3-dev libssl-dev flex
# 1.2 Download the source code of PETSc
cd ~
git clone -b release https://gitlab.com/petsc/petsc.git
cd petsc
# 1.3 Configure PETSc
./configure --with-debugging=0 --COPTFLAGS="-O3" --CXXOPTFLAGS="-O3" --FOPTFLAGS="-O3" --with-64-bit-indices=0 --download-hypre=1 --download-mumps=1 --download-scalapack=1 --download-metis=1 --download-parmetis=1 --download-bison=1 --download-ptscotch=1 --download-superlu_dist=1 --with-scalar-type=real
# 1.4 Make and install PETSc
./configure   --prefix=$HOME/petsc-install
make PETSC_DIR=$HOME/petsc PETSC_ARCH=arch-linux-c-opt all
make PETSC_DIR=$HOME/petsc PETSC_ARCH=arch-linux-c-opt install
# 1.5 Setup the environmental variables
export PETSC_DIR=$HOME/petsc
export PETSC_ARCH=arch-linux-c-opt
export LD_LIBRARY_PATH=$PETSC_DIR/$PETSC_ARCH/lib:$LD_LIBRARY_PATH
export PATH=$PETSC_DIR/$PETSC_ARCH/bin:$PATH
# Add the above command lines to the file ~/.bashrc. Then
source ~/.bashrc
# 1.6 Validate the installation
ls ~/petsc-install # Find if there exists lib/, include/ and so on.

# Step 2: Download the source code zip file
cd /path/you/want/to/put/zip file  # e.g. ~/PACS 
wget https://github.com/dealii/dealii/releases/download/v9.5.2/dealii-9.5.2.tar.gz
tar -xf dealii-9.5.2.tar.gz

# Step 3: Create and enter build directory
cd ~/PACS/dealii-9.5.2
mkdir build
cd build

# Step 4: Run cmake and make
cmake ../ -DCMAKE_INSTALL_PREFIX=~/PACS/dealii-install -DDEAL_II_WITH_PETSC=ON -DDEAL_II_WITH_MPI=ON -DDEAL_II_WITH_P4EST=ON -DDEAL_II_WITH_TRILINOS=OFF
make -j$(nproc)  # Use all the processes to compile

# Step 5: Install deal.ii
make install  # This will install deal.ii into ~/PACS/dealii-install

# Notice: remember to change the CMakeLists.txt file in order to run my project using deal.ii
# Deal.II configuration
find_package(deal.II 9.5.0
  COMPONENTS PETSc
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../dealii-install $ENV{DEAL_II_DIR}
)
# Please change ../dealii-install to the path where your dealii is installed really
```
