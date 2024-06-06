from collections import OrderedDict
from typing import List

import cv2
import numpy as np
import torch
from einops import rearrange, repeat
from gym import spaces
from mbodied_agents.agents.motion.rt1.tokenizers.action_tokenizer import RT1ActionTokenizer
from mbodied_agents.agents.motion.rt1.tokenizers.utils import batched_space_sampler, np_to_tensor
from mbodied_agents.agents.motion.rt1.transformer_network import TransformerNetwork
from mbodied_agents.base.agent import Agent
from mbodied_agents.base.motion import Motion
from mbodied_agents.types.controls import HandControl, JointControl, Pose
from mbodied_agents.types.vision import SupportsImage

observation_space = spaces.Dict(
    {
        "image_primary": spaces.Box(low=0.0, high=1.0, shape=(3, 224, 224), dtype=np.float32),
        "natural_language_embedding": spaces.Box(low=-np.inf, high=np.inf, shape=[512], dtype=np.float32),
    },
)
action_space_dict = OrderedDict(
    [
        (
            "xyz",
            spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
        ),
        (
            "rpy",
            spaces.Box(low=-np.pi, high=np.pi, shape=(3,), dtype=np.float32),
        ),
        (
            "grasp",
            spaces.Box(low=0, high=1.0, shape=(1,), dtype=np.float32),
        ),
    ],
)


class RT1Agent(Agent):
    """RT1Agent class responsible for interacting with the environment based on the agent's policy.

    This agent uses a TransformerNetwork to generate actions based on the given observations
    and context (natural language embedding).

    Attributes:
        config (dict): Configuration dictionary containing parameters for the agent.
        device (torch.device): The device to run the computations (CPU or CUDA).
        model (TransformerNetwork): The neural network model used for predicting actions.
        policy_state (Optional[torch.Tensor]): The internal state of the policy network.
        action_tokenizer (RT1ActionTokenizer): The action tokenizer for converting output to actions.
        image_history (List[torch.Tensor]): History of the past observations.
        step_num (int): Keeps track of the number of steps taken by the agent.
    """

    def __init__(self, config, weights_path: str = None, **kwargs) -> None:
        """Initializes the RT1Agent with the provided configuration and model weights.

        Args:
            config (dict): Configuration parameters for setting up the agent.
            weights_path (str, optional): Path to the pre-trained model weights. Defaults to None.
            **kwargs: Additional keyword arguments.

        Example:
            config = {
                "observation_history_size": 6,
                "future_prediction": 6,
                "token_embedding_dim": 512,
                "causal_attention": True,
                "num_layers": 6,
                "layer_size": 512,
            }
            agent = RT1Agent(config, weights_path="path/to/weights.pth")
        """
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransformerNetwork(
            observation_history_length=config["observation_history_size"],
            future_prediction_length=config["future_prediction"],
            token_embedding_dim=config["token_embedding_dim"],
            causal_attention=config["causal_attention"],
            num_layers=config["num_layers"],
            layer_size=config["layer_size"],
            observation_space=observation_space,
            action_space=spaces.Dict(action_space_dict),
            image_keys=["image_primary"],
            context_key="natural_language_embedding",
        ).to(self.device).eval()

        self.policy_state = None
        self.action_tokenizer = RT1ActionTokenizer(
            action_space=action_space_dict)

        if weights_path:
            self.model.load_state_dict(torch.load(
                weights_path, map_location=self.device))
        self.image_history = []
        for _i in range(6):
            self.image_history.append(torch.zeros(
                size=(224, 224, 3), dtype=torch.float, device=self.device))

        self.step_num: int = 0

    def act(self, 
        instruction_emb: torch.Tensor,
        image: SupportsImage,
    ) -> List[Motion]:
        """Generate a sequence of actions based on the provided instruction embedding and image.

        This method processes the image, maintains image history, constructs observations, and generates actions
        using the model. The actions include the hand's pose and grasp control.

        Args:
            instruction_emb (torch.Tensor): A tensor representing the natural language instructions.
            image (SupportsImage): An image used to inform the action decision.

        Returns:
            List[Motion]: A list of generated motions, each containing pose and grasp control.

        Example:
            >>> instruction_emb = torch.rand((1, 512))
            >>> image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            >>> agent = RT1Agent(config={'observation_history_size': 6, 'future_prediction': 6, 'token_embedding_dim': 512, 'causal_attention': True, 'num_layers': 6, 'layer_size': 512})
            >>> actions = agent.act(instruction_emb, image)
            >>> all(isinstance(action, HandControl) for action in actions)
            True
        """
        image = cv2.resize(np.array(image, dtype=np.uint8), (224, 224))
        self.image_history.append(torch.tensor(
            image / 255.0, dtype=torch.float32, device=self.device).permute(1, 0, 2))
        if len(self.image_history) > 6:
            self.image_history.pop(0)
        elif len(self.image_history) < 6:
            for _ in range(6 - len(self.image_history)):
                self.image_history.append(
                    torch.tensor(image / 255.0, dtype=torch.float32,
                                device=self.device).permute(1, 0, 2),
                )

        images = torch.stack(self.image_history)[None]

        video = rearrange(images.to(self.device), "b f h w c -> b f c h w")
        self.policy_state = np_to_tensor(
            batched_space_sampler(
                self.model.state_space,
                batch_size=1,
            ),
        )

        obs = {
            "image_primary": video,
            "natural_language_embedding": repeat(instruction_emb, "b c -> (6 b) c"),
        }

        outs, network_state = self.model(
            obs,
            self.policy_state,
        )
        out_tokens = outs[:, : (5 + 1), :, :].detach().cpu().argmax(dim=-1)
        self.out_tokens = out_tokens

        self.policy_state = network_state

        outs = self.action_tokenizer.detokenize(out_tokens)
        actions = [
            HandControl(
                pose=Pose(
                    x=outs["xyz"][0][i][0],
                    y=outs["xyz"][0][i][1],
                    z=outs["xyz"][0][i][2],
                    roll=outs["rpy"][0][i][0],
                    pitch=outs["rpy"][0][i][1],
                    yaw=outs["rpy"][0][i][2],
                ),
                grasp=JointControl(value=outs["grasp"][0][i]),
            )
            for i in range(6)
        ]

        self.step_num += 1
        return actions

if __name__ == "__main__":
    import doctest
    doctest.testmod()
