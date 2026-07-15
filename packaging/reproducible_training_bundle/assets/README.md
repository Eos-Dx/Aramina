# Aramis Docker Reproducible Training Bundle

Status: research-draft decision-support prototype.

## Windows

1. Install Docker Desktop once.
2. In Docker Desktop settings, enable the WSL 2 Linux engine.
3. Extract the entire ZIP to a local drive with sufficient free space.
4. Double-click `install_and_train.bat`, or run:

   ```powershell
   .\install_and_train.ps1
   ```

The script verifies the bundled H5 checksum, loads the bundled Linux runtime
image on first use, and runs preprocessing plus training. It does not install
Conda, Git, Python, pyFAI, Aramis, or XRD-preprocessing on Windows.

The H5 archive is mounted read-only from `data/combined_archive.h5`; it is
never copied into the Docker image. Generated artifacts and logs are written
to `outputs/` beside this README.

If Docker Desktop is not running, start it and wait until its Linux engine is
ready before rerunning the script.

## macOS/Linux

Run:

```bash
./install_and_train.sh
```

Docker is required on these systems as well.
