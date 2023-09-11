docker build . -t play_image:1.0 -f Dockerfile --cpuset-cpus 0-2
docker build . -t play_image_dev:1.0 -f Dockerfile_dev --cpuset-cpus 0-2
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
