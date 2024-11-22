from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from omegaconf import OmegaConf

from common.layers import PixelPreprocess, ShiftAug
from common.world_model import WorldModel


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
		nn.ConvTranspose2d(32, num_channels, 3, stride=1),
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
        if len(episodes) == 4:
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
    wm.requires_grad_(False)

    # Disable the shift augmentations
    wm.eval()

    print("Successfully created world model")


    # 512 hidden state
    # 32 x 4 x 4 unflattened.
    # encoder channels is 32
    decoder_channels = 64
    decoder = create_decoder(512, 3, decoder_channels)
    optimizer = torch.optim.Adam(decoder.parameters())
    # decoded = decoder(torch.randn((1, 512)))
    # print(decoded.shape)

    bsz = 32
    for epoch in range(500):
        for ep in range(len(episodes)):
            pointer = 0
            # optionally, here you can set batch size to be the length of the episode
            while pointer < len(episodes[ep]['obs']):
                # Nx9x64x64
                obs_batch = episodes[ep]['obs'].squeeze(1)[pointer:pointer+bsz]
                enc = wm.encode(obs_batch, task=None)
                decoded = decoder(enc)
                # Note: the size is 9x63x63, instead of the 9x64x64 that it should be.
                # I think this is because of truncation during stride or etc.
                # print(decoded.shape)

                # print(enc.shape, decoded.shape, obs_batch.shape, pointer, pointer+bsz)

                # compare decoded result with obs.
                # we normalize the images to the range [-0.5, 0.5]
                reconstruction_loss = F.mse_loss((obs_batch[:, -3:, :-1, :-1].float()/255.0) - 0.5, decoded)
                
                optimizer.zero_grad()
                reconstruction_loss.backward()
                optimizer.step()

                pointer += bsz

        print(reconstruction_loss.item())

        # save a decoded result and see what it looks like.
        if (epoch+1)%10 == 0:
            plt.rcParams['figure.figsize'] = [20, 4]

            obs = obs_batch
            for i in range(10):
                plt.subplot(2, 10, 1 + i)
                plt.imshow(obs[i, -3:, :-1, :-1].permute(1, 2, 0).detach().cpu().numpy())
                plt.title(f"Step {i} Obs")

                plt.subplot(2, 10, 11 + i)
                plt.imshow(decoded[i, -3:].permute(1, 2, 0).detach().cpu().numpy()+0.5)
                plt.title(f"Step {i} Reconstruction")

            plt.tight_layout()
            plt.savefig(f"comparison_{epoch+1}.png")
    

if __name__ == '__main__':
    main()
