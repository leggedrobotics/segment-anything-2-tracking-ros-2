


# segment-anything-2 real-time
Run Segment Anything Model 2 on a **live video stream** with ros2 wrapper. Based on https://github.com/Gy920/segment-anything-2-real-time


## Getting Started


### Docker:
In the root directory of the repository, run:
- build: ``` docker build -t sam2_rt -f docker/Dockerfile . ```

- run:  
``` 
    cd docker    
    docker compose run -e HOST_USERNAME=$(whoami) --service-ports sam2_rt
```

You may change the entrypoint for your desired application.

## Run:
python3 ros2/image_receiver.py

Click on the object you want to be tracked and let's go

## References:

- SAM2 Repository: https://github.com/facebookresearch/segment-anything-2
- SAM2 Real time: https://github.com/Gy920/segment-anything-2-real-time
