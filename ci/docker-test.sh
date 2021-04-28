#!/bin/bash
docker run -it -v "$PWD":/vaex -w /vaex ubuntu:trusty ci/runs-in-docker.sh
