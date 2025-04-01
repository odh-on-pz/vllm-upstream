#!/bin/bash

# This script build the CPU docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -e

export PIP_EXTRA_INDEX_URL=https://${ARTIFACTORY_USER}:${ARTIFACTORY_TOKEN}@na.artifactory.swg-devops.com/artifactory/api/pypi/sys-linux-power-team-ftp3distro-odh-pypi-local/simple
export TRUSTED_HOST=na.artifactory.swg-devops.com

# Setup cleanup
remove_docker_container() { docker rm -f cpu-test-ubi9-ppc || true; docker system prune -f; }
trap remove_docker_container EXIT
remove_docker_container

# Try building the docker image
docker build -t cpu-test-ubi9-ppc -f docker/Dockerfile.ppc64le .

# Run the image, setting --shm-size=4g for tensor parallel.
docker run -itd --entrypoint /bin/bash -v /tmp/:/root/.cache/huggingface --privileged=true --network host -e HF_TOKEN --name cpu-test-ubi9-ppc cpu-test-ubi9-ppc

function cpu_tests() {
  
  # offline inference
  docker exec cpu-test-ubi9-ppc bash -c "
    set -e
    python examples/offline_inference/basic.py"

  # Run basic model test
  docker exec cpu-test-ubi9-ppc bash -c "
    pip show setuptools 
    which vllm &&  /usr/local/bin/vllm --version

    dnf install gcc gcc-c++ gcc-gfortran libsndfile -y
    pip install pytest pytest-asyncio einops peft Pillow  soundfile transformers_stream_generator
    python -m pip install matplotlib==3.10.0 llvmlite==0.44.0 numba==0.61.0 --extra-index-url $PIP_EXTRA_INDEX_URL --trusted-host="$TRUSTED_HOST"
    pip install sentence-transformers librosa datamodel_code_generator
    pytest -v -s tests/models/embedding/language/test_cls_models.py::test_classification_models[float-jason9693/Qwen2.5-1.5B-apeach]
    pytest -v -s tests/models/embedding/language/test_embedding.py::test_models[half-BAAI/bge-base-en-v1.5]
    pytest -v -s tests/models/encoder_decoder/language -m cpu_model"
}

# All of CPU tests are expected to be finished less than 40 mins.
export -f cpu_tests
timeout 90m bash -c cpu_tests
