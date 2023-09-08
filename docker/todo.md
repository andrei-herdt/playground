- [ ] Mount host directory for vim plugins (see overlayfs)
```dockerfile
   mount -t overlay overlay -o lowerdir=/opt/rootfs/,upperdir=/workdir/armchroot-upper/,workdir=/workdir/armchroot-work/ /workdir/armchroot
```

- [ ] Add ohmyzsh to dockerfile_dev
- [ ] MarkdownPreview doesn't work from within docker. Port is not exposed?
