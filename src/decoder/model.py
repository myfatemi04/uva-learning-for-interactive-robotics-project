import torch
import torch.nn as nn
from common.layers import ShiftAug, PixelPreprocess
from pathlib import Path
from common.world_model import WorldModel
from omegaconf import OmegaConf

def get_sizes(in_shape, num_channels, act=None):
    """
    Basic convolutional encoder for TD-MPC2 with raw image observations.
    4 layers of convolution with ReLU activations, followed by a linear layer.
    """
    assert in_shape[-1] == 64 # assumes rgb observations to be 64x64
    layers = [
        ShiftAug(), PixelPreprocess(),
        nn.Conv2d(in_shape[0], num_channels, 7, stride=2), nn.ReLU(inplace=True),
        nn.Conv2d(num_channels, num_channels, 5, stride=2), nn.ReLU(inplace=True),
        nn.Conv2d(num_channels, num_channels, 3, stride=2), nn.ReLU(inplace=True),
        nn.Conv2d(num_channels, num_channels, 3, stride=1), nn.Flatten()
    ]
    
    print(nn.Sequential(*layers)(torch.randn((1, *in_shape))).shape)
    print(nn.Sequential(*layers[:-1])(torch.randn((1, *in_shape))).shape)
    print(nn.Sequential(*layers[:-3])(torch.randn((1, *in_shape))).shape)
    print(nn.Sequential(*layers[:-5])(torch.randn((1, *in_shape))).shape)
    print(nn.Sequential(*layers[:-7])(torch.randn((1, *in_shape))).shape)
    print(nn.Sequential(*layers[:-9])(torch.randn((1, *in_shape))).shape)

# The decoder can be of any kind.
# In this case, I train one that is symmetric to the encoder.
def create_decoder(in_dim, out_channels, num_channels, act=None):
	"""
	Basic convolutional decoder for TD-MPC2 with raw image observations.
	4 layers of convolution with ReLU activations, followed by a linear layer.
	"""
	# assert in_shape[-1] == 64 # assumes rgb observations to be 64x64

    # 3x64x64 -> c x 64x64

    # Replace each layer with an approximate inverse.
	layers = [
		nn.ConvTranspose2d(num_channels, out_channels, 7, stride=2), nn.ReLU(inplace=True),
		nn.ConvTranspose2d(num_channels, num_channels, 5, stride=2), nn.ReLU(inplace=True),
		nn.ConvTranspose2d(num_channels, num_channels, 3, stride=2), nn.ReLU(inplace=True),
		nn.ConvTranspose2d(num_channels, num_channels, 3, stride=1),
        # nn.Flatten()
        nn.Unflatten(-1, (32, 4, 4))
    ]
	if act:
		layers.append(act)
	return nn.Sequential(*layers[::-1])

def main():
    online_model_base = Path("logs/v4/hopper-hop/1/default")

    # Simple Prediction
    episodes = []
    for episodefile in (online_model_base / "episodes").iterdir():
        episodes.append(torch.load(episodefile))
        break

    # Load previously saved model.
    # One thing I notice: The Q function parameters 
    model = torch.load(online_model_base / "latest.pt")
    # print(model['model']['_target_Qs.params.9'].shape)

    # load the world model
    # (note: I should make it significantly easier to load the world model...)
    cfg = OmegaConf.load(online_model_base / ".hydra" / "config.yaml")
    cfg.obs_shape = {"rgb": (9, 64, 64)}
    cfg.action_dim = 4
    cfg.multitask = False
    cfg.task_dim = 0
    wm = WorldModel(cfg)
    wm.load_state_dict(model['model'])

    print("Successfully created world model")

    print(episodes[0].keys())
    # Nx9x64x64
    obs = episodes[0]['obs'].squeeze(1)
    enc = wm._encoder(obs)
    print(enc.shape)

    # get_sizes((9, 64, 64), 32)
    # 512 hidden state
    # 32 x 4 x 4 unflattened.
    # decoder = create_decoder(512, 9, 32)
    # decoded = decoder(torch.randn((1, 512)))
    # print(decoded.shape)

    # Note: the size is 9x63x63, instead of the 9x64x64 that it should be.

if __name__ == '__main__':
    main()
