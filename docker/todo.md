- [ ] git integration in vim inside docker
- [ ] MarkdownPreview doesn't work from within docker. Port is not exposed?
- [ ] In nvim, search function jumps automatically to normal mode when not found, creating havoc
- [ ] Download mujoco binaries https://github.com/google-deepmind/mujoco/releases/download/2.3.7/mujoco-2.3.7-linux-x86_64.tar.gz
- [ ] from vim to tmux jump doesn't work
install j
- [ ] Something blocks run_docker. Pressing ctrl-c helps
- [ ] Store history outside of docker
- [ ] git public key not available inside docker
- [ ] dubious access rights error message in docker
- [ ] What is the short key for closing buffer
- [ ] Make lazygit work in docker
- [ ] Integrate clangformat into vim
- [ ] Add xdg-open

- [ ] Mount host directory for vim plugins (see overlayfs)
- [ ] Add surround plugin
- [ ] **tab to call fzf
- [ ] make jumping to file from editor work
- [ ] how to find any file
```dockerfile
   mount -t overlay overlay -o lowerdir=/opt/rootfs/,upperdir=/workdir/armchroot-upper/,workdir=/workdir/armchroot-work/ /workdir/armchroot
```

