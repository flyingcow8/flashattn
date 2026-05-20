Building FlashAttention2 from source code on MACA
--------------------------------------------------


# Create conda environment based on MACA

## Step 1: install miniconda virtual environment
- Download and install miniconda package of verison py38, e.g. Miniconda3-py38_23.11.0-2-Linux-x86_64.sh on linux x86_64 platform.

- Configure pip
```
    pip config set global.index-url http://your_website_to_source/r/pypi/simple
    pip config set install.trusted-host your_website_to_source
```

- Configure channels
```
    conda config --add channels http://your_website_to_source/r/tsinghua-conda-pkgs/main/
    conda config --add channels http://your_website_to_source/r/tsinghua-conda-pkgs/free/
    conda config --add channels http://your_website_to_source/r/tsinghua-conda-cloud/conda-forge/
    conda config --add channels http://your_website_to_source/r/tsinghua-conda-cloud/msys2/
    conda config --add channels http://your_website_to_source/r/tsinghua-conda-cloud/pytorch
    conda config --set show_channel_urls yes
    conda config --show channels
    sed -i.bak '/- defaults/d' ~/.condarc
    conda config --remove channels defaults
```

## Step 2: create conda virtual environment
```
    conda install python=3.8.16
    conda create -n your_conda python=3.8.16
    conda activate your_conda
```

## Step 3: install MACA PyTorch
```
    pip install your_maca_torch.whl --force-reinstall --no-deps
```

## Step 4: install other dependent packages
```
    pip install ninja
    pip install einops
    pip install setuptools==78.1.1
    pip install pytest
    pip install packaging
    pip install SentencePiece
    pip install accelerate
    pip install wheel
```

## Step 5: set MACA environment
```
    export MACA_PATH=/your/maca/path
    export MACA_CLANG_PATH=$MACA_PATH/mxgpu_llvm/bin
    export CUDA_PATH=$MACA_PATH/tools/cu-bridge
    export LD_LIBRARY_PATH=$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:$MACA_PATH/ompi/lib:$LD_LIBRARY_PATH
```

## Step 6：build FlashAttention2
Build flash-attn in FlashAttention2 diretory with the following command and the whl package will be in dist directory:
```
    python setup.py bdist_wheel
```

## Step 7：install flash-attn whl package
```
    pip install dist/flash-attn-version.whl
```
