import json

import torch


class ActionTokenizer:
    def __init__(self):
        """Initializes the ActionTokenizer.

        Loads the ra_to_token_map from a JSON file and sets the total number of bins for discretization.
        """
        self.bins = 256
        with open('mbodied/agents/motion/rt_pali/action_tokenizer/ra_to_token_map.json', 'r') as f:
            self.ra_to_token_map = json.load(f)

    def discretize_values(self, pose_data: dict) -> dict:
        """Discretizes continuous pose data into discrete bins.

        Args:
            pose_data (dict): Dictionary containing pose data with keys like 'grasp', 'terminated', 
                              and other positional and orientation values that need to be discretized.

        Returns:
            dict: Dictionary containing the discretized pose data.
        """
        discrete_data = {}

        for key, scaled_value in pose_data.items():
            if key == 'grasp' or key == 'terminated':
                # Ensure grasp is binary: 0 or 1
                bin_index = (self.bins-1) if scaled_value > 0.5 else 0
                discretized = f"ra_{bin_index}"
                discrete_data[key] = discretized
            else:
                # Quantize the scaled value to a bin index
                quantized_tensor = torch.quantize_per_tensor(
                    torch.tensor([scaled_value], dtype=torch.float32),
                    scale=1/(self.bins - 1),
                    zero_point=0,
                    dtype=torch.quint8
                )
                bin_index = int(quantized_tensor.int_repr().item())
                discretized = f"ra_{bin_index}"
                discrete_data[key] = discretized

        return discrete_data

    def reverse_discretize_values(self, discrete_data: dict) -> dict:
        """Converts discrete values back to continuous values.

        Args:
            discrete_data (dict): Dictionary containing discretized pose data.

        Returns:
            dict: Dictionary containing the continuous pose data.
        """
        inverse_data = {}

        for key, token in discrete_data.items():
            bin_index = int(token.rsplit('_', 1)[1])
            scaled_value = bin_index / (self.bins - 1)

            inverse_data[key] = scaled_value

        return inverse_data

    def tokenize(self, pose_data: dict) -> str:
        """Converts pose data into a string of action tokens.

        Args:
            pose_data (dict): Dictionary containing pose data to be tokenized.

        Returns:
            str: A space-separated string of action tokens.
        """
        discretized_data = self.discretize_values(pose_data)
        action_tokens = list(discretized_data.values())
        return " ".join([self.ra_to_token_map[ra_tkn] for ra_tkn in action_tokens])

    def detokenize(self, tokens: list) -> dict:
        """Converts a list of action tokens back into continuous pose data.

        Args:
            tokens (list): List of action tokens.

        Returns:
            dict: Dictionary containing the continuous pose data.
        """
        token_to_ra_map = {v: k for k, v in self.ra_to_token_map.items()}
        action_tokens = [token_to_ra_map[tkn] for tkn in tokens]
        discretized_data = dict(zip(["terminated", "x", "y", "z", "roll", "pitch", "yaw", "grasp"],
                                    action_tokens)
                                )
        return self.reverse_discretize_values(discretized_data)
