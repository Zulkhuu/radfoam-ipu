# Radiant Foam renderer on Graphcore IPU

Implementation of **[Radiant Foam: Real-Time Differentiable Ray Tracing](https://radfoam.github.io/)** on Graphcore IPUs.  
This repository provides an **IPU-accelerated renderer** designed to leverage Graphcore’s hardware for efficient real-time differentiable ray tracing. It includes the renderer itself, scene preprocessing tools, and utilities for running and visualizing results.

![Radiant Foam — garden](./assets/garden_rgb_image.png)

## What’s Inside

- **IPU Renderer** (C++): The core renderer that builds and runs inside a Docker image on a cloud machine equipped with Graphcore IPUs.  
- **Scene Partitioning** (Python): A set of utilities and Jupyter notebooks for preprocessing scenes and saving partitioned outputs.  

### Also Required
- **Remote Viewer** (C++): A lightweight client that runs on your laptop or workstation and streams frames from the renderer in real time.  
  The Remote Viewer can be installed from the official repository: [remote_render_ui](https://github.com/Zulkhuu/remote_render_ui).  



## Quick start

### 1) Build the IPU renderer (cloud machine with IPUs)
```bash
git clone --recursive https://github.com/Zulkhuu/radfoam-ipu.git
cd radfoam-ipu/docker
./build.sh
./run.sh
```

`run.sh` starts the container and attaches a shell.  
Inside the Docker environment, the repository is mounted, and your current working directory is set to the repository root.  

From inside the container, run:

```bash
mkdir build && cd build
cmake ..
make -j
```

### 2) Build the remote viewer (your local machine, separate repo)
```bash
git clone --recursive https://github.com/Zulkhuu/remote_render_ui.git
mkdir -p remote_render_ui/build && cd remote_render_ui/build
cmake -G Ninja ..
ninja -j16
```

### 3) Prepare data

Precomputed partitions for MipNeRF360 and Deep Blending are available here:  
<https://drive.google.com/drive/folders/1Ld3mUZJYZW05Yrm6Jb_VFWbP3qg8kyE4?usp=sharing>

#### Optional: Run scene partitioning

Scene partitioning require a Python 3 environment. Ensure that Python 3 is installed, then install the required dependencies:

```bash
cd scene_partitioning
pip install -r requirements.txt
```

Once setup is complete, open and run the notebook: **[scene_partitioning.ipynb](./scene_partitioning.ipynb)** to partition a Radiant Foam scene and save the results.  

## Running the demo

#### Start the renderer on cloud machine:
```
./main -i <input_file> --port 5000
```

#### Run the remote viewer on your local machine:
```
./remote-ui -w 1600 -h 1100 --port 5000
```

By default, the system uses **port 5000** for communication. If you want to use a different port: 
- Set it in `run.sh`.  
- Make sure to use the same port when executing both the **IPU Renderer** and the **Remote Viewer**.  

## Acknowledgements

This work is based on the paper:
> Shrisudhan Govindarajan, Daniel Rebain, Kwang Moo Yi, Andrea Tagliasacchi.  
> **Radiant Foam: Real-Time Differentiable Ray Tracing.**

It builds on and uses the following projects:
- [3D Gaussian Splatting on Graphcore IPUs](https://github.com/nmjfry/gaussian_splat_ipu)
- [Remote User Interface (remote viewer)](https://github.com/Zulkhuu/remote_render_ui)

The following public libraries are included via Git submodules under the external/ directory and are fetched when cloning recursively:

- **GLM** — OpenGL Mathematics library  
  `external/glm` → <https://github.com/g-truc/glm>

- **packetcomms** — Lightweight packet-based communications helpers  
  `external/packetcomms` → <https://github.com/mpups/packetcomms>

- **videolib** — Utilities for video/stream handling  
  `external/videolib` → <https://github.com/markp-gc/videolib>