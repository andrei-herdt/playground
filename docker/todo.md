- [ ] Mount host directory for vim plugins (see overlayfs)
```dockerfile
   mount -t overlay overlay -o lowerdir=/opt/rootfs/,upperdir=/workdir/armchroot-upper/,workdir=/workdir/armchroot-work/ /workdir/armchroot
```

- [ ] Add ohmyzsh to dockerfile_dev
- [ ] git integration in docker
- [ ] MarkdownPreview doesn't work from within docker. Port is not exposed?
- [ ] Search jumps automatically to normal mode, creating havoc
- [ ] Download mujoco binaries https://github.com/google-deepmind/mujoco/releases/download/2.3.7/mujoco-2.3.7-linux-x86_64.tar.gz
- [ ] pip install robot_descriptions
