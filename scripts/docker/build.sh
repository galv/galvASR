#!/bin/bash

REPO_DIR=$(git rev-parse --show-toplevel)

# We need to mount the repo in the container's file system so that it
# can access all the submodules.
docker build -t galvasr -v "$REPO_DIR":/galvASR/ DefaultBuild

#nvidia-docker run --name galvASR_container1 -ti galvASR bash
