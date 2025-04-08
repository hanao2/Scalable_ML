import yaml
from ml_collections import ConfigDict
from utils import set_XLA_flags
from utils import Batch, TrainState,
from utils import mixed_precision, activation_checkpointing, gradient_accumulation

set_XLA_flags()

if __name__ == "__main__":
    with open("../utils/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    config = ConfigDict(config["network_size"])
    #mixed_precision(samples=100, config=config)
    #activation_checkpointing(samples=100, config=config, remat=False)
    # for num_minibatches in [1, 4, 8, 16]:
    #    print(f'Number of minibatches: {num_minibatches}')
    #    gradient_accumulation(samples=16, config=config, num_minibatches=num_minibatches, dropout_rate=0.0)
