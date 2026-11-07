"""Resolve Qwen3 RoPE parameters across Transformers configuration versions."""
import math
import numbers


def resolve_qwen3_rope_theta(config):
    parameters = getattr(config, 'rope_parameters', None)
    if isinstance(parameters, dict) and 'rope_theta' in parameters:
        theta = parameters['rope_theta']
    else:
        theta = getattr(config, 'rope_theta', 1000000)
    if isinstance(theta, bool) or not isinstance(theta, numbers.Real):
        raise ValueError('Qwen3 rope_theta must be a positive finite number')
    if not math.isfinite(theta) or theta <= 0:
        raise ValueError('Qwen3 rope_theta must be a positive finite number')
    return theta
