


# segment-anything-2 real-time
Run Segment Anything Model 2 on a **live video stream** with ros2 wrapper. Based on https://github.com/Gy920/segment-anything-2-real-time


## Getting Started


### Docker:
- build: docker build -t sam2_rt -f Dockerfile .
- run: docker run --gpus all -it -e HOST_USERNAME=$(whoami) -e HOST_UID=$(id -u) -e HOST_GID=$(id -g) -v /home/$(whoami):/home/$(whoami) sam2_rt

## Run

python ros2/image_receiver.python

Click on the object you want to be tracked and let's go

### Installation

```bash
pip install -e .
```
### Download Checkpoint

Then, we need to download a model checkpoint.

```bash
cd checkpoints
./download_ckpts.sh
```

Then SAM-2-online can be used in a few lines as follows for image and video and **camera** prediction.


## References:

- SAM2 Repository: https://github.com/facebookresearch/segment-anything-2
- SAM2 Real time: https://github.com/Gy920/segment-anything-2-real-time
# segement-anything-2-real-time-ros-2

run using docker compose: docker-compose run -e HOST_USERNAME=$(whoami) --service-ports  sam2_rt