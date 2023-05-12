from dataclasses import dataclass
from torch.distributed.fsdp import (
    ShardingStrategy,
    BackwardPrefetch,
    StateDictType
)

import torch


@dataclass
class train_config:
    # mixed precision
    use_mixed_precision: bool = False
    use_fp16: bool = False
    # sharding_strategy: ShardingStrategy = ShardingStrategy.HYBRID_SHARD
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD
    fsdp_activation_checkpointing: bool = True
    checkpoint_type = StateDictType.SHARDED_STATE_DICT