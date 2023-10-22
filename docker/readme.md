
sudo groupadd docker
sudo usermod -aG docker $USER

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo chmod 666 /var/run/docker.sock

# Run build_images.sh
./build_image.sh

# Generate ssh key on host
ssh-keygen -t ed25519 -C "your_email@example.com"

# Copy public key to github

# Set vim config for docker
cd $PLAYGROUND_HOME/docker/workdir/.config
sudo git clone git@github.com:andrei-herdt/neovim-config.git

# Nvidia
For NVIDIA support you need to install `nvidia-docker`: [installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker). 

> Note if running ./run_docker.sh gives you error:
> ```
> docker: Error response from daemon: could not select device driver "" with capabilities: gpu.
> ```
> This means you haven't properly configured the docker support for your graphics card.
