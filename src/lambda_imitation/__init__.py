"""lambda-imitation: IQ-Learn (Inverse Q-Learning) imitation learning.

Public API re-exports from the three sub-modules.
"""

from lambda_imitation.buffer import (
    Buffer,
    BufferFunctions,
    BufferSample,
    create_buffer,
    create_sample,
)
from lambda_imitation.iqlearn import (
    Head,
    Hyperparameters,
    IQLearnFunctions,
    IQLearnGraphs,
    IQLearnState,
    MLPFeatureExtractor,
    NetworkGraphs,
    NetworkState,
    TwinCriticState,
    create_iqlearn,
    extract_buffer_shapes,
)
from lambda_imitation.utils import (
    EnvSpec,
    create_iqlearn_from_env,
    env_spec_from_gymnasium,
    env_spec_from_gymnax,
    env_spec_from_jumanji,
)

__all__ = [
    # buffer
    "Buffer",
    "BufferFunctions",
    "BufferSample",
    "create_buffer",
    "create_sample",
    # iqlearn
    "Head",
    "Hyperparameters",
    "IQLearnFunctions",
    "IQLearnGraphs",
    "IQLearnState",
    "MLPFeatureExtractor",
    "NetworkGraphs",
    "NetworkState",
    "TwinCriticState",
    "create_iqlearn",
    "extract_buffer_shapes",
    # utils
    "EnvSpec",
    "create_iqlearn_from_env",
    "env_spec_from_gymnasium",
    "env_spec_from_gymnax",
    "env_spec_from_jumanji",
]
