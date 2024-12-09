1. start devcontainer 
    - `ctrl + shift + p`
    - Select Dev Container: Reopen in Container

2. To change the terminal shell from sh to bash
```
bash
```

3. Setup environment
```
mkdir glomap/external
cd glomapexternal
wget https://download.pytorch.org/libtorch/cu124/libtorch-shared-with-deps-2.5.1%2Bcu124.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu121.zip
cd ../..
```

4. build glomap
check cuda version from https://developer.nvidia.com/cuda-gpus.
i.e. NVIDIA RTX A5000 is compute capability 86. 

Change line 8 of [CMakeLists.txt](CMakeLists.txt) to the corresponding computer capability.
```
DCMAKE_CUDA_ARCHITECTURES=86 # from native to 86.
```

```
mkdir build
cd build
cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=86 -DCTESTS_ENABLED=ON
ninja
```