# Docker

This directory contains the `Dockerfile` we use to run our models on the GPU servers. As recommended by the chair, we

- use a Base-Image by NVIDIA, which already includes most complex libraries,
- use pinned versions within our `requirements.txt`, and
- don't include our source code in the Image but rather mount it via a Docker Volume.

## DockerHub

The Image is available at the following URL: https://hub.docker.com/r/benjaminfrost99/reddit_impostor
