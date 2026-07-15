# Aramis Docker Reproducible Training Bundle

Status: research-draft decision-support prototype.

## Windows

1. Extract the entire ZIP to a local drive with sufficient free space.
2. Double-click `install_and_train.bat`, or run:

   ```powershell
   .\install_and_train.ps1
   ```

On its first use, the script downloads and installs Docker Desktop with its
WSL 2 Linux backend, then starts the Docker engine. It requires internet
access and accepts the Docker license. If Windows needs a restart after WSL 2
is enabled, restart and run the same file again. Later runs reuse Docker.

The script verifies the bundled H5 checksum, loads the bundled native
`linux/amd64` runtime image on first use, and runs preprocessing plus training.
It does not install Conda, Git, Python, pyFAI, Aramis, or XRD-preprocessing on
Windows.

The H5 archive is mounted read-only from `data/combined_archive.h5`; it is
never copied into the Docker image. Generated artifacts and logs are written
to `outputs/` beside this README.

If Docker Desktop already exists but its Linux engine is stopped, the script
starts it and waits up to five minutes for it to become ready.

## macOS/Linux

Run:

```bash
./install_and_train.sh
```

Docker is required on these systems as well.

The macOS/Linux launcher automatically loads the native `linux/arm64` image on
Apple Silicon and `linux/amd64` image on Intel/AMD systems.
