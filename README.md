# PACSProject

This is a project for the PACS course.  
Its research topic is accelerating AMG by deep learning method.

## Project structure

```
|--PACSProject
    |--include
        |--*.hpp
    |--src
        |--*.cpp
    |--model
        |--model.py
    |--data
        |--data_processing.py
    |--main.py
```

## Dataset generation
If you want to generate the train and test datasets for different differential models, please read the following instructions. 
At first, you should ensure that you have installed the package `deal.ii` and `PETSc`. I will give an instruction as the following: 
```bash
# Step 1: Install PETSc
sudo apt install libpetsc-dev petsc-dev

# Step 2: Download the source code zip file
cd /path/you/want/to/put/zip file  # e.g. ~/APSC 
wget https://github.com/dealii/dealii/releases/download/v9.5.2/dealii-9.5.2.tar.gz
tar -xf dealii-9.5.2.tar.gz

# Step 3: Create and enter build directory
cd ~/APSC/dealii-9.5.2
mkdir build
cd build

# Step 4: Run cmake and make
cmake ../ \
  -DCMAKE_INSTALL_PREFIX=~/APSC/dealii-install \
  -DDEAL_II_WITH_PETSC=ON \
  -DDEAL_II_INSTALL_TEMPLATES=ON
make

# Step 5: Install deal.ii
make install  # This will install deal.ii into ~/APSC/dealii-install

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

