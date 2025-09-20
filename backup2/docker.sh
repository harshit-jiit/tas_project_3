nvidia-docker run -it --gpus device=7 --rm --name jiit_204170001 \
  --shm-size=32g --ipc=host \
  -v /home/jiit_2404170001/Downloads/job1/gpu-dev-container/:/workspace/2404170001 \
  -v $(pwd)/workspace:/workspace/project \
  -p 2032:8888 
  2404170001-gpu-dev

  