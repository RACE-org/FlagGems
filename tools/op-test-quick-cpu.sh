#!/bin/bash

pytest -s tests/test_blas_ops.py --ref=cpu  &> blas.log
pytest -s tests/test_reduction_ops.py --ref=cpu --mode=quick &> reduction_ops.log
pytest -s tests/test_general_reduction_ops.py --ref=cpu --mode=quick &> general_reduction_ops.log
pytest -s tests/test_norm_ops.py --ref=cpu --mode=quick &> norm_ops.log
pytest -s tests/test_unary_pointwise_ops.py --ref=cpu --mode=quick &> unary_pointwise_ops.log
pytest -s tests/test_binary_pointwise_ops.py --ref=cpu --mode=quick &> binary_pointwise_ops.log
pytest -s tests/test_special_ops.py --ref=cpu --mode=quick &> special_ops.log
pytest -s tests/test_tensor_constructor_ops.py --ref=cpu --mode=quick &> tensor_constructor_ops.log
pytest -s tests/test_attention_ops.py --ref=cpu &> attention.log
