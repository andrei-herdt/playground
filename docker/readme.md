docker build . -t play_image:1.0 -f Dockerfile --cpuset-cpus 0-2
docker build . -t play_image_dev:1.0 -f Dockerfile_dev --cpuset-cpus 0-2

sudo groupadd docker
sudo usermod -aG docker $USER

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo chmod 666 /var/run/docker.sock


# Set vim config for docker
cd $PLAYGROUND_HOME/docker/workdir/.config
sudo git clone https://github.com/andrei-herdt/neovim-config.git nvim
