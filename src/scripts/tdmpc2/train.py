import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["LAZY_LEGACY_OP"] = "0"
import shutil
import warnings

warnings.filterwarnings("ignore")
import hydra
import torch
from termcolor import colored

from common.buffer import Buffer
from common.logger import Logger
from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_env
from tdmpc2 import TDMPC2
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer

torch.backends.cudnn.benchmark = True


def get_sub_state_dict(state_dict, key):
    return {
        state_dict_key[len(key + '.'):]: state_dict_value
        for (state_dict_key, state_dict_value) in state_dict.items()
        if state_dict_key.startswith(key + '.')
    }

@hydra.main(config_name="config", config_path=".")
def train(cfg: dict):
    """
    Script for training single-task / multi-task TD-MPC2 agents.

    Most relevant args:
            `task`: task name (or mt30/mt80 for multi-task training)
            `model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
            `steps`: number of training/environment steps (default: 10M)
            `seed`: random seed (default: 1)

    See config.yaml for a full list of args.

    Example usage:
    ```
            $ python train.py task=mt80 model_size=48
            $ python train.py task=mt30 model_size=317
            $ python train.py task=dog-run steps=7000000
    ```
    """
    assert torch.cuda.is_available()
    assert cfg.steps > 0, "Must train for at least 1 step."
    assert (cfg.eval_episodes >= cfg.num_envs) and (
        cfg.eval_episodes % cfg.num_envs == 0
    ), "Number of eval episodes must be divisible by the number of envs."

    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    print(colored("Work dir:", "yellow", attrs=["bold"]), cfg.work_dir)

    # Store the hydra config...
    shutil.copytree(str(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir) + "/.hydra", str(cfg.work_dir) + "/.hydra")

    # The environment must be initialized before the TDMPC2 object, because
    # it populates the obs_shape key in the cfg object.
    env = make_env(cfg)
    agent = TDMPC2(cfg)

    '''
    ADDITIONAL FUNCTIONALITY FOR TESTING CROSS-TASK GENERALIZATION.
    
    Initialize world model encoder and dynamics from checkpoint, if checkpoint exists.
    OPTIONS:
    1) `encoder_and_dynamics_checkpoint` (str)
        The path to the .pt file to use, or a folder containing 'latest.pt'.

    2) `encoder_and_dynamics_freeze` (bool)
        Whether to freeze the _encoder and _dynamics attributes of the TDMPC2 model.

    '''
    if cfg.encoder_and_dynamics_checkpoint:
        path = str(cfg.encoder_and_dynamics_checkpoint)
        if '.pt' not in path:
            path = os.path.join(path, 'latest.pt')
        
        # Load the 'latest.pt' file.
        # For the format, see `tdmpc2.py` => .save() method.
        # (I use self.agent.save()).
        # This will return the weights from the WorldModel object,
        # stored as the 'model' attribute in the TDMPC2 object.
        data = torch.load(path)
        state_dict = data['model']

        # Extract the state_dict pertaining to the _encoder and _dynamics keys.
        agent.model._encoder.load_state_dict(
            get_sub_state_dict(state_dict, '_encoder')
        )
        agent.model._dynamics.load_state_dict(
            get_sub_state_dict(state_dict, '_dynamics')
        )
        agent.model._infer_action.load_state_dict(
            get_sub_state_dict(state_dict, '_infer_action')
        )

        if cfg.encoder_and_dynamics_freeze:
            agent.model._encoder.requires_grad_(False)
            agent.model._dynamics.requires_grad_(False)
            agent.model._infer_action.requires_grad_(False)

    trainer_cls = OfflineTrainer if cfg.multitask else OnlineTrainer
    trainer = trainer_cls(
        cfg=cfg,
        env=env,
        agent=agent,
        buffer=Buffer(cfg),
        logger=Logger(cfg),
    )
    trainer.train()
    print("\nTraining completed successfully")


if __name__ == "__main__":
    train()
