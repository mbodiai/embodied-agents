from collections import OrderedDict

import numpy as np
from absl import flags
from gym import spaces
from lightning.pytorch import LightningModule
from tokenizers.action_tokenizer import RTX1ActionTokenizer as ActionTokenizer

from rt1.transformer_network import TransformerNetwork

FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", None, "Random seed.")
flags.DEFINE_bool("data_augmentation", True, "Whether or not to use data augmentation.")
flags.DEFINE_bool("random_erasing", False, "Random image shape masking.")

# Checkpoint flags
flags.DEFINE_string("checkpoint_path", None, "Path to checkpoint or 'last'.")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Checkpoint directory to save to.")
flags.DEFINE_string('resume', '', 'Resume [all, dataloader]')
flags.DEFINE_integer("checkpoint_frequency", 100, "Checkpoint frequency in steps.")
flags.DEFINE_integer("log_frequency", 50, "Log frequency in steps.")
flags.DEFINE_bool("log_images", False, "Log images.")
flags.DEFINE_integer("log_image_frequency", 5000, "Log image frequency in steps.")

# Model flags
flags.DEFINE_string('model', 'rt1', 'Model [rt1, rtx1]')
flags.DEFINE_integer('observation_history_size', 6, 'Observation history size')
flags.DEFINE_integer('future_action_window_size', 5, 'Future action window size')
flags.DEFINE_bool('causal_attention', True, 'Causal attention')
flags.DEFINE_integer('summary_depth', 1, 'Summary depth')

#RT1 flags
flags.DEFINE_integer('token_embedding_dim', 512, 'Token embedding dimension')
flags.DEFINE_integer('num_layers', 8, 'Number of layers')
flags.DEFINE_integer('layer_size', 128, 'Layer size')
flags.DEFINE_integer('num_heads', 8, 'Number of heads')
flags.DEFINE_float('dropout_rate', 0.1, 'Dropout rate')
flags.DEFINE_integer('image_tokens_size', 8, 'Image tokens size')

class RT1Module(LightningModule):

  def __init__(self, config=None):
    super().__init__(config)
    self.action_tokenizer = ActionTokenizer(dataset_name=config['oxe_datasets'])
    self.configure_model()

  def configure_model(self) -> None:
    if hasattr(self, 'model'):
      return
    observation_space = spaces.Dict({
        'image_primary':
            spaces.Box(low=0.0, high=1.0, shape=(3, 224, 224), dtype=np.float32),
        'natural_language_embedding':
            spaces.Box(low=-np.inf, high=np.inf, shape=[512], dtype=np.float32)
    })
    action_space_dict = OrderedDict([
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
            spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        ),
    ])
    if self.config['norm_actions'] == 'across_actions':
      action_space_dict['mean'] = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
      action_space_dict['std'] = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)

    action_space = spaces.Dict(action_space_dict)

    action_mean = None
    action_std = None
    
    self.model = TransformerNetwork(
        observation_history_length=self.config['observation_history_size'],
        future_prediction_length=self.config['future_action_window_size'],
        token_embedding_dim=self.config['token_embedding_dim'],
        causal_attention=self.config['causal_attention'],
        num_layers=self.config['num_layers'],
        layer_size=self.config['layer_size'],
        observation_space=observation_space,
        action_space=action_space,
        image_keys=['image_primary'],
        context_key='natural_language_embedding',
        action_mean=action_mean,
        action_std=action_std,
    )

  #@profile
  def training_step(self, batch, batch_idx):
    pass

def default_config():
  abls_flags = FLAGS.get_flags_for_module(__name__)
  return {abs_flag.name: abs_flag.value for abs_flag in abls_flags}
