#!/bin/bash

pytest -s tests/test_blas_ops.py &> ../logs/reduction_ops_0520.log
pytest -s tests/test_reduction_ops.py --ref=cpu --mode=quick &> ../logs/reduction_ops_0520.log
pytest -s tests/test_general_reduction_ops.py --ref=cpu --mode=quick &> ../logs/general_reduction_ops_0520.log
pytest -s tests/test_norm_ops.py --ref=cpu --mode=quick &> ../logs/norm_ops_0520.log
pytest -s tests/test_unary_pointwise_ops.py --ref=cpu --mode=quick &> ../logs/unary_pointwise_ops_0520.log
pytest -s tests/test_binary_pointwise_ops.py --ref=cpu --mode=quick &> ../logs/binary_pointwise_ops_0520.log
pytest -s tests/test_special_ops.py --mode=quick &> ../logs/special_ops_0520.log
pytest -s tests/test_tensor_constructor_ops.py --mode=quick &> ../logs/tensor_constructor_ops_0520.log
pytest -s tests/test_attention_ops.py --mode=quick &> ../logs/attention_ops_0520.log
