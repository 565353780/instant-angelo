cd ..
git clone --recursive https://github.com/nvlabs/tiny-cuda-nn

pip install ninja

pip3 install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu124

pip install matplotlib opencv-python imageio imageio-ffmpeg \
  scipy CuMCubes pyransac3d torch_efficient_distloss \
  tensorboard click open3d trimesh pymcubes plyfile

pip install pytorch-lightning <2
pip install omegaconf==2.2.3
pip install nerfacc==0.3.3

cd tiny-cuda-nn
cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo -j
cd bindings/torch
python setup.py install
