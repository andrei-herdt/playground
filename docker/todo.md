- [ ] install killall
- [ ] Make lazygit work in docker
- [ ] Download mujoco binaries https://github.com/google-deepmind/mujoco/releases/download/2.3.7/mujoco-2.3.7-linux-x86_64.tar.gz
install j
- [ ] Integrate clangformat into vim
- [ ] Change theme of tmux and vim inside docker conveniently

- [ ] Mount host directory for vim plugins (see overlayfs)
- [ ] Add surround plugin
- [ ] make jumping to file from editor work
- [ ] how to find any file
```dockerfile
   mount -t overlay overlay -o lowerdir=/opt/rootfs/,upperdir=/workdir/armchroot-upper/,workdir=/workdir/armchroot-work/ /workdir/armchroot
```
- [ ] from vim to tmux jump doesn't work

