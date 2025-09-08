# Radiant Foam renderer implementation on Graphcore IPU

IPU implementation of a paper [Radiant Foam: Real-Time Differentiable Ray Tracing](https://radfoam.github.io/) by Shrisudhan Govindarajan, Daniel Rebain, Kwang Moo Yi2, and Andrea Tagliasacchi.

![image](./assets/garden_rgb_image.png)

## Installation

The demo consists of two parts: IPU based renderer that runs on cloud and remote local viewer.

### Build the renderer on cloud machine with IPU

Build Docker image by running:
```
git clone --recursive https://github.com/Zulkhuu/radfoam-ipu.git
cd docker
./build.sh
```

Start Docker container by:
```
cd docker
./run.sh
```
It will start running docker and attach to the terminal

#### Clone and build the renderer
```
git clone --recursive https://github.com/Zulkhuu/radfoam-ipu.git
mkdir radfoam-ipu/build
cd radfoam-ipu/build
cmake ..
make
```

### On your local laptop or workstation:

#### Clone and build the remote viewer
```
git clone --recursive https://github.com/Zulkhuu/remote_render_ui.git
mkdir remote_render_ui/build
cd remote_render_ui/build
cmake -G Ninja ..
ninja -j16
```

## Data
Scene partitioned MipNeRF360 dataset files can be downloaded from [here](https://drive.google.com/drive/folders/1Ld3mUZJYZW05Yrm6Jb_VFWbP3qg8kyE4?usp=sharing)

## Running the demo

#### Start the renderer on cloud machine:
```
./main -i <input_file>
```

#### Run the remote viewer on your local machine:
```
./remote-ui -w 1600 -h 1100 --port 5000
```

This works builds on and uses following repositories
 - [3D Gaussian Splatting on Graphcore IPUs](https://github.com/nmjfry/gaussian_splat_ipu)
 - [Remote User Interface](https://github.com/nmjfry/remote_render_ui)
