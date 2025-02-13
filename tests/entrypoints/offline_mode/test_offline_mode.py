# SPDX-License-Identifier: Apache-2.0
"""Tests for HF_HUB_OFFLINE mode"""
import importlib
import sys

import pytest
import urllib3

from vllm import LLM
from vllm.distributed import cleanup_dist_env_and_memory

MODEL_CONFIGS = [
    {
        "model": "facebook/opt-125m",
        "enforce_eager": True,
        "gpu_memory_utilization": 0.20,
        "max_model_len": 64,
        "max_num_batched_tokens": 64,
        "max_num_seqs": 64,
        "tensor_parallel_size": 1,
    },
    {
        "model": "mistralai/Mistral-7B=B-Instruct-v0.1",
        "enforce_eager": True,
        "gpu_memory_utilization": 0.95,
        "max_model_len": 64,
        "max_num_batched_tokens": 64,
        "max_num_seqs": 64,
        "tensor_parallel_size": 1,
        "tokenizer_mode": "mistral",
    },
    {
        "model": "sentence-transformers/all-MiniLM-L12-v2",
        "enforce_eager": True,
        "gpu_memory_utilization": 0.20,
        "max_model_len": 64,
        "max_num_batched_tokens": 64,
        "max_num_seqs": 64,
        "tensor_parallel_size": 1,
    },
]


@pytest.fixture(scope="module")
def cache_models():
    # Cache model files first
    for model_config in MODEL_CONFIGS:
        LLM(**model_config)
        cleanup_dist_env_and_memory()

    yield


@pytest.mark.skip_global_cleanup
@pytest.mark.usefixtures("cache_models")
def test_offline_mode(monkeypatch):
    # Set HF to offline mode and ensure we can still construct an LLM
    try:
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        monkeypatch.setenv("VLLM_NO_USAGE_STATS", "1")

        def disable_connect(*args, **kwargs):
            raise RuntimeError("No http calls allowed")

        monkeypatch.setattr(urllib3.connection.HTTPConnection, "connect",
                            disable_connect)
        monkeypatch.setattr(urllib3.connection.HTTPSConnection, "connect",
                            disable_connect)

        # Need to re-import huggingface_hub and friends to setup offline mode
        _re_import_modules()
        # Cached model files should be used in offline mode
        for model_config in MODEL_CONFIGS:
            LLM(**model_config)
    finally:
        # Reset the environment after the test
        # NB: Assuming tests are run in online mode
        monkeypatch.delenv("HF_HUB_OFFLINE")
        monkeypatch.delenv("VLLM_NO_USAGE_STATS")
        _re_import_modules()
        pass


def _re_import_modules():
    hf_hub_module_names = [
        k for k in sys.modules if k.startswith("huggingface_hub")
    ]
    transformers_module_names = [
        k for k in sys.modules if k.startswith("transformers")
        and not k.startswith("transformers_modules")
    ]

    reload_exception = None
    for module_name in hf_hub_module_names + transformers_module_names:
        try:
            importlib.reload(sys.modules[module_name])
        except Exception as e:
            reload_exception = e
            # Try to continue clean up so that other tests are less likely to
            # be affected

    # Error this test if reloading a module failed
    if reload_exception is not None:
        raise reload_exception
