- [ ] Mount host directory for vim plugins (see overlayfs)
```dockerfile
   mount -t overlay overlay -o lowerdir=/opt/rootfs/,upperdir=/workdir/armchroot-upper/,workdir=/workdir/armchroot-work/ /workdir/armchroot
```

- [ ] Add ohmyzsh to dockerfile_dev
- [ ] git integration in docker
- [ ] MarkdownPreview doesn't work from within docker. Port is not exposed?
- [ ] In nvim, search function jumps automatically to normal mode when not found, creating havoc
- [ ] Download mujoco binaries https://github.com/google-deepmind/mujoco/releases/download/2.3.7/mujoco-2.3.7-linux-x86_64.tar.gz
- [ ] pip install robot_descriptions
- [ ] from vim to tmux jump doesn't work
git config --global core.editor vi
git config ...
start tmux automa
install j

- [ ] Something blocks run_docker. Pressing ctrl-c helps
- [ ] Entry point should be /workdir/playground/
- [ ] Store history outside of docker
