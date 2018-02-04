# detectron_service

The detailled version of https://hackernoon.com/how-to-use-detectron-facebooks-free-platform-for-object-detection-9d41e170bbcb

## OS

I've used Ubuntu 16.04 on top of a [G2-15](https://www.ovh.com/fr/public-cloud/instances/tarifs/) OVH virtual machine.

## Dependencies

```sh
sudo apt-get install cmake build-essential python-pip
```


### CUDA

Get CUDA installer from [NVIDIA](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1704&target_type=deblocal) page.

### NNPack

```sh
pip install PeachPy
git clone https://github.com/Maratyszcza/NNPACK.git
cd NNPACK
mkdir build
cmake -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE .. && make
sudo make install
```

### cuDNN

Register/login on https://developer.nvidia.com/cudnn and download 'cuDNN v7.0.5 Runtime Library for Ubuntu16.04 (Deb)' and 'cuDNN v7.0.5 Developer Library for Ubuntu16.04 (Deb)'

```sh
sudo dpkg -i libcudnn*.deb
```


### Caffe2

```sh
sudo apt-get install libeigen3-dev libleveldb-dev libopencv-dev librocksdb-dev libgflags-dev python3-dev liblmdb-dev libsnappy-dev libgoogle-glog-dev libopenmpi-dev
sudo apt-get install python-numpy python-setuptools
```

```sh
pip install future six protobuf
```

```sh
git clone --recursive https://github.com/caffe2/caffe2.git && cd caffe2
make && cd build && sudo make install
python -c 'from caffe2.python import core' 2>/dev/null && echo "Success" || echo "Failure"
```

### Cocoapi

```sh
sudo pip install Cython
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
sudo make install
```

### Detectron

```sh
sudo pip install numpy pyyaml matplotlib opencv-python>=3.0 setuptools Cython mock scipy
```

```sh
git clone https://github.com/facebookresearch/Detectron.git && cd Detectron
cd lib && make && cd ..
```


## Detectron web service

```sh
git clone https://github.com/ticapix/detectron_service.git && cd detectron_service
make install
make run
```

# Usage

```sh
curl -v -F image=@ovh_parking.jpg http://detectron.ticapix.ovh:8080/analyse -o test.png
```
