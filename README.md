# Flysense Jetson Camera

Easy GPU and HW Accelerators API.

jetson-utils

```bash
sudo apt-get install freeglut3-dev libglew-dev mesa-utils libpython3-dev python3-numpy
git clone https://github.com/dusty-nv/jetson-utils
cd jetson-utils && git checkout 35593c5ad2c1c62f6f0166d6581945fb58fd1f7b
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j4
sudo make install
```

Tested on:

- TX2 Developer Kit
- AGX Xavier

```shell
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build .

sudo make install
# sudo make uninstall
```
