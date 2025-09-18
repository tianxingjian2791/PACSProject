# PACSProject

This is a project for the PACS course.  
Its research topic is accelerating AMG by deep learning method.

## Project structure

```
|--PACSProject
    |--include
        |--DiffusionModel.hpp
        |--ElasticModel.hpp
        |--Pooling.hpp
        |--StokesModel.hpp
    |--src
        |--main.cpp
    |--model
        |--cnn_model.py
        |--gat_model.py
    |--data
        |--cnn_data_processing.py
        |--gat_data_processing.py
    |--weights
        |--*.pth
    |--train.py
    |--val.py
    |--train_cnn_log.txt
    |--train_gat_log.txt
```

## Dataset generation
If you want to generate the train and test datasets for different differential models, please read the following instructions. 
At first, you should ensure that you have installed the package `deal.ii` and `PETSc`. I will give an instruction as the following: 
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
Now we can generate the datasets. 
```
# generate the datasets of Diffusion Problem
$ build/PACSProject D train
$ build/PACSProject D test

# generate the datasets of Elastic Problem
$ build/PACSProject E train
$ build/PACSProject E test

# generate the datasets of Stokes Problem
$ build/PACSProject S train
$ build/PACSProject S test
```

