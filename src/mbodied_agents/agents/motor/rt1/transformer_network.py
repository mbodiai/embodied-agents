# Subject to the terms and conditions of the Apache License, Version 2.0 that the original code follows,
# I have retained the following copyright notice written on it.

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# You can find the original code from here[https://github.com/google-research/robotics_transformer].
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from gym import spaces
from mbodied_agents.agents.motor.rt1.tokenizers import action_tokenizer

from .tokenizers.image_tokenizer import RT1ImageTokenizer
from .transformer import Transformer


class TransformerNetwork(nn.Module):
    """A transformer based actor network."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Dict, 
        image_keys: List[str] = ['image_primary'],
        context_key: str = 'natural_language_embedding',
        vocab_size: int = 256, # Token dimension.
        num_heads: int = 8,
        image_tokens_size: int = 8,
        token_embedding_dim: int = 512, # Embedded token dwimension.
        num_layers: int = 1,
        layer_size: int = 4096,  # Attention key_dim which is the size of each attention head for query, key and values.
        dropout_rate: float = 0.1,
        use_token_learner: bool = True,
        observation_history_length: int = 6,
        future_prediction_length: int = 1, # Must be <= observation_history_length.
        causal_attention: bool = True,
    ):
        super().__init__()

        
        self.observation_space = observation_space
        self.action_space = action_space
        self.observation_history_length = observation_history_length
        self.future_prediction_length = future_prediction_length
        self.causal_attention = causal_attention
        self.vocab_size = vocab_size
        self.context_key = context_key
        self.image_keys = image_keys
        self.num_heads = num_heads
        self.use_token_learner = use_token_learner
        self.image_tokens_size = image_tokens_size

        self.loss_object = nn.CrossEntropyLoss(reduction="none")
        self.image_tokenizers = nn.ModuleDict({
            key:  RT1ImageTokenizer(
                embedding_output_dim=token_embedding_dim,
                language_embedding_size=token_embedding_dim,
                use_token_learner=use_token_learner,
                num_tokens=self.image_tokens_size,
            )
            for key in self.image_keys
        })
        self.action_tokenizer = action_tokenizer.RT1ActionTokenizer(
            action_space, vocab_size=self.vocab_size,
        )
        self.transformer = Transformer(
            num_layers=num_layers,
            layer_size=layer_size,
            num_heads=num_heads,
            feed_forward_size= token_embedding_dim,
            dropout_rate=dropout_rate,
            vocab_size=vocab_size,
            input_token_emb_dim=token_embedding_dim
        )

        # Get the number of tokens
        self.tokens_per_action = self.action_tokenizer.tokens_per_action
        self.tokens_per_context_image = self.image_tokenizers[
            self.image_keys[0]
        ].tokens_per_context_image
        self.token_embedding_dim = token_embedding_dim

        # generate loss mask and attention mask
        self.default_attention_mask = torch.as_tensor(self.generate_masks())

        # this is used only when random sampling
        # when sampling, the output is used as network_state
        self.state_space = spaces.Dict(
            {
                "context_image_tokens": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(
                        observation_history_length,
                        self.tokens_per_context_image,
                        token_embedding_dim,
                    ),
                    dtype=np.float32,
                ),
                # "context_pos_orn": spaces.Box(
                #     low=-np.inf,
                #     high=np.inf,
                #     shape=(observation_history_length, self.tokens_per_context_image),
                #     dtype=np.float32,
                # ),
                "action_tokens": spaces.MultiDiscrete(
                    np.full((observation_history_length, self.tokens_per_action), vocab_size)
                ),
                # Stores where in the window we are.
                # This value is within range [0, observation_history_length + 1].
                # When seq_idx == observation_history_length, context_image_tokens and
                # action_tokens need to be shifted to the left.
                "seq_idx": spaces.Discrete(observation_history_length + 1)
                # Our data is like context_image_tokens + action_tokens + context_image_tokens + action_tokens + context_image_tokens ...
                # 1 time step means [context_image_tokens + action_tokens]
                # seq_idx means which time steps we are. But it is adjusted to observation_history_length when it exceeds observation_history_length.
            }
        )

    def get_action_index_for_token(self, k):
        """Returns action associated with the token at given position `k`.

        If k is not an action token then it returns -1.
        If k is part of the first action in the sequence then returns 0 etc.

        Args:
            k: an int that represents the position in the sequence.

        Returns:
            The index of the action that this position belongs to, or if this
            position is part of an image token then returns -1.
        """
        if k < 0 or k >= self.all_num_tokens:
            return -1

        n = k
        if (
            n % self.single_time_step_num_tokens < self.tokens_per_context_image
        ):  # check whether k is context_image token
            return -1
        return int(
            n / self.single_time_step_num_tokens
        )  # return which time index that k belongs to.

    # _action_tokens_mask is for loss computing. This has all indexes of action tokens in all tokens.
    # We can know which output tokens are action predictions by _action_tokens_mask - 1.
    # _default_attention_mask is modified causaul mask because we will only use observations tokens when predicting actions.
    # So we also have to mask action tokens.
    def generate_masks(self):
        """Generate mask for action prediction loss and attention visualization."""
        # each time step = [image, action]
        self.single_time_step_num_tokens = (
            self.tokens_per_action + self.tokens_per_context_image
        )

        # full sequence = [prefix context + N x timestep + postfix context]
        self.all_num_tokens = (
            self.observation_history_length * (self.single_time_step_num_tokens)
        )

        # create mask for action predition loss
        # self.action_tokens_mask has all indexes of action tokens in all tokens.
        self.action_tokens_mask = []
        for n in range(0, self.all_num_tokens, self.single_time_step_num_tokens):
            for x in range(0, self.tokens_per_action, 1):
                self.action_tokens_mask.append(x + n + self.tokens_per_context_image)

        default_attention_mask =   np.ones((self.all_num_tokens, self.all_num_tokens), dtype=int)
        if self.causal_attention:
            # The look ahead mask ensures causality.
            # This is a lower triangular matrix. All elements other than 0 are 1.
            # 0 means mask.
            default_attention_mask = np.tril(
                default_attention_mask
            )

        action_mask = np.ndarray(shape=(self.all_num_tokens, self.all_num_tokens), dtype=int)


        for i in range(self.all_num_tokens):
            for j in range(self.all_num_tokens):
                action_i = self.get_action_index_for_token(i)
                action_j = self.get_action_index_for_token(j)
                mask = 0
                if (
                    action_i != -1 and action_j != -1
                ):  # Check both of i and j are actions.
                    # Ignore actions of previous time steps.
                    if self.causal_attention and action_j < action_i:
                        mask = 1
                    # If we're not auto-regression, ignore action of current time step.
                    if self.causal_attention and action_j == action_i and j <= i:
                        mask = 1
                action_mask[i, j] = mask
        default_attention_mask -= action_mask
        return default_attention_mask

    #@profile
    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        network_state: Dict[str, torch.Tensor],
        action_readout_tokens: Optional[torch.Tensor] = None,
    ):
        """Calls the transformer network.

        Args:
            observations: Observation data including image and natural language
                embedding in dict of Tensors.
        network_state: Only used for inference. Network state data including time seq idx, image tokens, action
            tokens.
        action_readout_tokens: Optional readout tokens to pass in. Input tokens will be zero otherwise

        Returns:
            A tuple `(Detokenized output actions, network state)`.
        """
        with torch.no_grad():
            outer_rank = self.get_outer_rank(observations)
            assert outer_rank in (1, 2), "outer rank should be 1 or 2"

            b, t = self.get_batch_size_and_seq_len(observations)
            # network_state is used when inference.
            # b : batch size
            # t: observation_history_length of this model

            # context_image_tokens: (b, t, num_tokens, embedding_dim)
            # action_tokens: (b, t, self.tokens_per_action)
        context_image_tokens, previous_action_tokens, attention_mask = self.get_tokens_and_mask(
            observations, network_state
        )
        if action_readout_tokens is None:
            action_readout_tokens = torch.zeros(
                (b, t, self.tokens_per_action, self.token_embedding_dim), dtype=torch.float).to(context_image_tokens.device)
        # print('\n\n\n action tokens: ', action_tokens.shape)
        # self.aux_info = {"action_labels": action_tokens}

        if outer_rank == 1:  # This is an inference call
            # run transformer in loop to produce action tokens one-by-one
            seq_idx = network_state["seq_idx"][0]
            action_t = torch.minimum(
                seq_idx, torch.tensor(self.observation_history_length - 1)
            )
            # Transformer shifts all to the left by one step by default (it's usually
            # predicting the next token as default training task...).
            transformer_shift = -1
            # We only want to get the action predicted at time_step.
            # This index means the output of the last observation token that is at action_t time step.
            start_index = (
                transformer_shift
                + self.tokens_per_context_image
                + action_t * (self.single_time_step_num_tokens)
            )
            current_action_tokens = []
            action_predictions_logits = []
            # Repeat inference tokens_per_action times.
            for k in range(self.tokens_per_action):
                action_index = start_index + k
                # token: (1, 1)
                # token_logits: (1, 1 vocab_size)
                token, token_logits = self.transformer_call_and_slice(
                    context_image_tokens,
                    action_readout_tokens,
                    attention_mask=attention_mask,
                    batch_size=b,
                    slice_start=action_index,  # slicing single action dimension
                )
                action_predictions_logits.append(token_logits)

                current_action_tokens.append(token)

                # Add the predicted token to previous_action_tokens.
                previous_action_tokens = previous_action_tokens.view(
                    b, -1
                )  # [b, t, self.tokens_per_action] -> [b, t * self.tokens_per_action]
                action_start_index = (action_t * self.tokens_per_action) + k
                # replace action_tokens[:, action_start_index] with the predicted token. Note that this is not insert.
                action_tokens = torch.concat(
                    [
                        previous_action_tokens[:, :action_start_index],
                        token,
                        previous_action_tokens[:, action_start_index + 1 :],
                    ],
                    dim=1,
                )
                action_tokens = action_tokens.view(
                    b, t, self.tokens_per_action
                )  # [b, t * self.tokens_per_action] -> [b, t, self.tokens_per_action]

            self.aux_info.update(
                {
                    # action_predictions_logits is
                    # [1, self.tokens_per_action, self.vocab_size]
                    "action_predictions_logits": torch.concat(
                        action_predictions_logits, 1
                    )
                }
            )

            predicted_tokens_for_output = torch.concat(
                current_action_tokens, 1
            )  # [1, self.tokens_per_action]
            one_state_action_tokens = predicted_tokens_for_output.unsqueeze(
                1
            )  # [1, 1, self.tokens_per_action]

            # Add predicted action tokens  to network_state['action_tokens']
            state_action_tokens = network_state[
                "action_tokens"
            ]  # (1, observation_history_length, self.tokens_per_action)
            # replace state_action_tokens[:, action_t, ...] with the predicted tokens. Note that this is not insert.
            network_state["action_tokens"] = torch.concat(
                [
                    state_action_tokens[:, :action_t, ...],
                    one_state_action_tokens,
                    state_action_tokens[:, action_t + 1 :, ...],
                ],
                dim=1,
            )

            # Increment the time_step for the next inference call.
            # network_state['seq_idx'] never exceed observation_history_length.
            network_state["seq_idx"] = torch.minimum(
                seq_idx + 1, torch.tensor(self.observation_history_length)
            )[None]

            self.loss = torch.tensor(0.0)
            output_actions = self.action_tokenizer.detokenize(predicted_tokens_for_output)
            return output_actions, network_state
        else:
            # training call --> simply run one transformer forward pass
            # output_tokens: (bs, t*num_tokens, vocab_size)
            output_tokens = self.transformer_call(
                context_image_tokens,
                action_readout_tokens,
                attention_mask=attention_mask,
                batch_size=b,
            )

            # Gather all predicted actions for the action loss. Use fancy index to extract all predicted actions.
            predicted_action_index = torch.tensor(self.action_tokens_mask) - 1
             # (bs, t*tokens_per_action, vocab_size)
            action_logits = output_tokens[
                :, predicted_action_index
            ] 
            # (bs, t, self.tokens_per_action, vocab_size)
            action_logits_for_training = action_logits.view(
                b, t, self.tokens_per_action, -1
            )  
            return action_logits_for_training[:,:self.future_prediction_length+1,:,:], network_state

            # Only take the last action as the action.
            # action_logits_for_output is [b, self.tokens_per_action, emb]
            # action_logits_for_output = action_logits_for_training[
            #     :, -1
            # ]  # This will take action at last time step in this training.
            # predicted_tokens_for_output is [b, self.tokens_per_action]
            # predicted_tokens_for_output = torch.argmax(action_logits_for_output, dim=-1)
            # action_logits_for_training = action_logits_for_training.permute(0, 3, 1, 2)
            # num_items = float(b * t) * self.single_time_step_num_tokens
            # # action_logits_for_training: (b, t, self.tokens_per_action, vocab_size)
            # # action_tokens, (b, t, self.tokens_per_action)
            # # action_loss: (b, t)
            # action_loss = self.loss_object(
            #     action_logits_for_training, action_tokens
            # )  # (b, t, self.tokens_per_action)

            # self.loss = action_loss

            # # store action labels and predictions for visualization
            # self.aux_info.update(
            #     {
            #         "action_predictions": torch.argmax(
            #             action_logits_for_training, dim=-1
            #         ),
            #         "action_loss": action_loss,
            #         "actor_loss_mask": torch.ones((b), dtype=torch.float32),
            #     }
            # )

        # output_actions: Dict[str, np.ndarray]


        # output_actions is the last actions.
        # network_stape is the past state that is used for next inference.

    def get_outer_rank(self, observations: Dict[str, torch.Tensor]) -> int:
        # used to determine training vs inference call
        # outer_rank will be 2 -> [b, t] during training and
        # outer_rank will be 1 -> [b] during inference

        for k in observations.keys():
            obs_value = observations[k]
            obs_value_shape = obs_value.shape

            obs_space = self.observation_space[k]
            obs_space_shape = obs_space.shape
            break
        return len(obs_value_shape) - len(obs_space_shape)

    def get_batch_size_and_seq_len(self, observations):
        image_shape = observations[self.image_keys[0]].shape
        b = image_shape[0]
        t = image_shape[1]
        return b, t

    #@profile
    def transformer_call(
        self,
        context_image_tokens: torch.Tensor,  # (b, t, num token, emb_dim)
        action_readout_tokens: torch.Tensor,  # (b, t, self.tokens_per_action, embed_dim)
        batch_size: int,
        attention_mask: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        input_token_sequence = self.assemble_input_token_sequence(
            context_image_tokens, action_readout_tokens
        ) 
        output_tokens, _ = self.transformer(
            input_token_sequence, attention_mask
        )  # (bs, t*num_tokens, vocab_size)
        return output_tokens

    # input_token_sequence = [context_image_tokens + action_tokens]
    def assemble_input_token_sequence(
        self, context_image_tokens, action_tokens
    ):

       
        input_token_sequence = torch.concat(
            (context_image_tokens, action_tokens), dim=2
        )

        input_token_sequence = rearrange(
            input_token_sequence, "b t n e -> b (t n) e"
        )
        return input_token_sequence

    # Call transformer, slice output, return predicted token.
    #@profile
    def transformer_call_and_slice(
        self, *args, slice_start: int = 0, slice_length: int = 1, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_tokens = self.transformer_call(*args, **kwargs)

        slice_end = slice_start + slice_length
        token_logits = output_tokens[
            :, slice_start:slice_end, :
        ]  # (b, slice_length, vocab_size)
        token = torch.argmax(token_logits, dim=-1)

        return token, token_logits

    #@profile
    def get_tokens_and_mask(
        self,
        observations: Dict[str, torch.Tensor],
        network_state: Dict[str, torch.Tensor],
    ):
        # tokenize all inputs
        context_image_tokens, _ = self.tokenize_images(
            observations, network_state
        )

        action_tokens = self.tokenize_actions(observations, network_state)

        attention_mask = self.default_attention_mask

        return (context_image_tokens, action_tokens, attention_mask)

    # At training, we don't use network_state at all.
    # At training, this will just convert image and context into tokens.
    #@profile
    def tokenize_images(self, observations, network_state):
         # [b, t, c, h, w] or [b, c, h, w]
        image = observations[self.image_keys[0]]
        outer_rank = self.get_outer_rank(observations)

        if outer_rank == 1:  # This is an inference call
             # 0 ~ observation_history_length
            seq_idx = network_state["seq_idx"][0] 
            time_step = torch.minimum(
                seq_idx, torch.tensor(self.observation_history_length - 1)
            )
            image = rearrange(image, "b c h w -> b 1 c h w")

        image_shape = image.shape
        b = image_shape[0]
        input_t = image_shape[1]
        c = image_shape[2]
        h = image_shape[3]
        w = image_shape[4]

        # return context from observation after check whether context is in observation.
        context = self.extract_context_from_observation(
            observations, input_t
        )  # [b, t, emb-size] or None


        image = image.view((b, input_t, c, h, w))

        # get image tokens
        context_image_tokens = []
        for i in range(len(self.image_keys)):
             # (batch, t, num_tokens, embedding_dim)
            context_image_tokens.append(
                self.image_tokenizers[self.image_keys[i]](image, context=context)
            ) 
        context_image_tokens = sum(context_image_tokens)

        # update network state at inference
        # At inference, we retain some context_image_tokens to accelerate computation.
        # At inference, context_image_tokens : (batch, 1, num_tokens, embedding_dim)
        # At inference, network_state stores context_image_tokens of past time steps.
        # Here, we combine past context_image_tokens of network_state with current context_image_tokens.
        # network_state only store tokens within observation_history_length time steps.
        # This means network_state does not store the tokens for all past steps, but only for observation_history_length time steps.
        # if current time step >= observation_history_length, we store context_image_tokens after we discard the oldest context_image_tokens.
        # Here, we implement that by shifting state_image_token to the left.
        if outer_rank == 1:  # This is an inference call
            state_image_tokens = network_state[
                "context_image_tokens"
            ]  # (1, observation_history_length, tokens_per_context_image, token_embedding_size)
            # network_state as input for this call is the output from the last call.
            # Therefore, we need to shift all images to the left by 1 in the time axis
            # to align with the time dim in this call.
            state_image_tokens = (
                torch.roll(state_image_tokens, -1, 1)
                if seq_idx == self.observation_history_length
                else state_image_tokens
            )
            # if seq_idx == observation_history_length, state_image_tokens will be shifted to the left along time axis
            # seq_idx will be incremented in forward function. But it is adjusted so that it never exceed observation_history_length.
            # Therefore, shiftimg will always occur when time step exceeds observation_history_length.
            # maximum of time_step is self.observation_history_length - 1
            context_image_tokens = torch.concat(
                [
                    state_image_tokens[:, :time_step, ...],
                    context_image_tokens,  # Note that in inference, size of context_image_tokens is (batch, 1, num_tokens, embedding_dim)
                    state_image_tokens[
                        :, time_step + 1 :, ...
                    ]  # if time_step == observation_history_lengths -1, this will be empty tensor.
                    # So this tensor will be ignored when concat
                ],
                dim=1,
            )

        network_state["context_image_tokens"] = context_image_tokens
        return context_image_tokens, network_state
    #@profile
    def tokenize_actions(self, observations, network_state):
        outer_rank = self.get_outer_rank(observations)

        if outer_rank == 1:  # This is an inference call
            action_tokens = network_state["action_tokens"]
            seq_idx = network_state["seq_idx"][0]
            # network_state as input for this call is the output from the last call.
            # Therefore, we need to shift all actions by 1 to the left.
            action_tokens = (
                torch.roll(action_tokens, -1, 1)
                if seq_idx == self.observation_history_length
                else action_tokens
            )
        else:
            assert outer_rank == 2
            # self.actions was set through set_actions function.
            if (
                not hasattr(self, 'actions') or self.actions is None
            ):  # When there is no action that will be tokenized to begin with, we create zero tensor.
                b, t = self.get_batch_size_and_seq_len(observations)
                action_tokens = torch.zeros(
                    (b, t, self.tokens_per_action), dtype=torch.int32
                )
            else:
                action_tokens = self.action_tokenizer.tokenize(self.actions)
        return action_tokens

    # output context from observation. size: [b, t, emb-size]
    #@profile
    def extract_context_from_observation(self, observations, seq_len):
        """Extract context from observation."""
        context = None
        if self.context_key is not None:
            outer_rank = self.get_outer_rank(observations)
             # [b, t, emb-size] or [b, emb-size]
            context = observations[
               self.context_key
            ] 
            if outer_rank == 1:
                context = torch.tile(context[:, None], [1, seq_len, 1])
                # [b, emb-size] ->  [b, 1, emb-size] -> [b, seq_len, emb-size]
        return context
    

    def get_actor_loss(self) -> torch.Tensor:
        return self.loss

    def get_aux_info(self) -> Dict[str, Any]:
        return self.aux_info


if __name__ == "__main__":
    net = TransformerNetwork()
