import json

import pytorch_lightning as pl
import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AdamW,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)

import wandb
from mbodied.agents.motion.rt_pali.action_tokenizer.action_tokenizer import ActionTokenizer

# Configuration for Low-Rank Adaptation (LoRA)
lora_config = LoraConfig(
    r=8,
    target_modules=[
        "q_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    task_type="CAUSAL_LM",
)


class VLAModel(pl.LightningModule):
    """Vision-Language Action Model for Conditional Generation using PaliGemma."""

    def __init__(self,
                 pretrained_model_name: str = 'google/paligemma-3b-mix-224',
                 learning_rate: float = 1e-4):
        """Initializes the VLAModel with a pre-trained PaliGemma model and processor.

        Args:
            pretrained_model_name (str): The name of the pre-trained model to use.
            learning_rate (float): The learning rate for the optimizer.
        """
        super().__init__()

        # Load pre-trained processor and model
        processor = PaliGemmaProcessor.from_pretrained(pretrained_model_name)
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            pretrained_model_name)

        # Freeze vision and projector parameters
        for param in model.vision_tower.parameters():
            param.requires_grad = False
        for param in model.multi_modal_projector.parameters():
            param.requires_grad = False

        # Apply LoRA configuration to model
        model = get_peft_model(model, lora_config)

        self.model = model
        self.processor = processor
        self.learning_rate = learning_rate

        # Load valid action token IDs
        valid_token_ids_file = 'mbodied/agents/motion/rt_pali/action_tokenizer/valid_tokens.json'  # noqa: S105
        with open(valid_token_ids_file, 'r') as f:
            valid_action_token_ids_dict = json.load(f)

        self.valid_action_token_ids = list(
            valid_action_token_ids_dict.values())
        self.action_tokenizer = ActionTokenizer()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            tokens (torch.Tensor): Tokenized inputs.

        Returns:
            torch.Tensor: Model outputs.
        """
        return self.model(**tokens)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Training step executed for each batch.

        Args:
            batch (dict): The batch of data.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: Loss value.
        """
        images = batch['image']
        language_instruction = batch['language_instruction']
        action_tokens = batch['action_tokens']

        tokens = self.processor(
            text=language_instruction,
            images=images,
            suffix=action_tokens,
            return_tensors="pt",
            padding="longest",
            tokenize_newline_separately=False,
        )
        tokens = tokens.to(self.device)

        outputs = self(tokens)
        loss = outputs.loss
        wandb.log({"train_loss": loss.item()})
        self.log("train_loss", loss.item())
        return loss

    def generate_action_map(self, image, task_instruction) -> str:
        """Generates an action map given an image and a task instruction.

        This method preprocesses the inputs, restricts the logits to valid action
        token IDs, and generates a set of action tokens that are then detokenized
        into a human-readable action map.

        Args:
            image (torch.Tensor): The input image tensor.
            task_instruction (str): The task instruction text.

        Returns:
            str: The generated action map as a string.
        """
        # Preprocess the image and task instruction
        inputs = self.processor(
            text=task_instruction,
            images=image,
            padding="longest",
            do_convert_rgb=True,
            return_tensors="pt",).to(self.device)

        def restrict_logits(_, scores):
            mask = torch.full(scores.shape, float('-inf'), device=self.device)
            # Only enable valid token indices
            mask[:, self.valid_action_token_ids] = 0
            return scores + mask

        output = self.model.generate(
            **inputs,
            max_new_tokens=8,
            min_new_tokens=8,
            logits_processor=[restrict_logits],
        )[:, -8:]

        action_tokens = [self.processor.decode(
            token, skip_special_tokens=True) for token in output[0]]

        return self.action_tokenizer.detokenize(action_tokens)

    def on_train_epoch_end(self) -> None:
        """Callback at the end of each training epoch to log training loss and generate action maps.

        This method prints the training loss for the current epoch and, if the epoch 
        is a multiple of a specified interval (here, 1), it generates action maps 
        for a sample batch from the validation data loader and logs them to W&B.

        Returns:
            None
        """
        # Print the training loss
        print(f'Training loss at epoch {self.current_epoch}:',
              self.trainer.callback_metrics.get('train_loss'))

        if self.current_epoch % 1 == 0:
            # For demonstration, we'll assume there's a method to get a sample batch
            sample_batch = next(iter(self.trainer.datamodule.val_dataloader()))

            # Use only the first item in the batch
            image = sample_batch["image"][0]
            task_instruction = sample_batch["language_instruction"][0]
            target_text = sample_batch["action_tokens"][0]

            target_actions = self.action_tokenizer.detokenize(
                target_text.split())

            generated_actions = self.generate_action_map(
                image, task_instruction)

            # Log the generated text and target
            wandb.log({
                "generated_actions": generated_actions,
                "target_actions": target_actions,
            })

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configures the optimizer for the training process.

        This method returns an AdamW optimizer with the specified learning rate for 
        the model parameters.

        Returns:
            torch.optim.Optimizer: The configured AdamW optimizer.
        """
        return AdamW(self.parameters(), lr=self.learning_rate)
