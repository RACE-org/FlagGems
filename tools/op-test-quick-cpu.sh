#!/bin/bash

pytest -s tests/test_blas_ops.py --ref=cpu --mode=quick && \
pytest -s tests/test_reduction_ops.py --ref=cpu --mode=quick && \
pytest -s tests/test_general_reduction_ops.py --ref=cpu --mode=quick && \
pytest -s tests/test_norm_ops.py --ref=cpu --mode=quick && \
pytest -s tests/test_unary_pointwise_ops.py --ref=cpu --mode=quick && \
pytest -s tests/test_binary_pointwise_ops.py --ref=cpu --mode=quick && \
pytest -s tests/test_special_ops.py --ref=cpu --mode=quick && \
pytest -s tests/test_tensor_constructor_ops.py --ref=cpu --mode=quick && \
pytest -s tests/test_attention_ops.py --ref=cpu --mode=quick
