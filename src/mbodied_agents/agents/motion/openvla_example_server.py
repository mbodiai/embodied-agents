import gradio as gr
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
from io import BytesIO


class OpenVLAInterface:
    """This class encapsulates the OpenVLA Agent's capabilities for remote action prediction."""

    def __init__(self, model_name="openvla/openvla-7b", device="cuda"):
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)

    def predict_action(self, image_base64, instruction, unnorm_key=None):
        """Predicts the robot's action based on the provided image and instruction."""
        image = Image.open(BytesIO(image_base64))
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self.processor(prompt, image).to("cuda", dtype=torch.bfloat16)
        action = self.model.predict_action(
            **inputs, unnorm_key=unnorm_key, do_sample=False)
        return action


def create_interface():
    """Creates and returns a Gradio Interface for the OpenVLA robot action prediction."""
    vla_interface = OpenVLAInterface()
    gr_interface = gr.Interface(
        fn=vla_interface.predict_action,
        inputs=[
            gr.Textbox(label="Base64 Image"),
            gr.Textbox(label="Instruction"),
            gr.Textbox(label="Unnorm Key"),
        ],
        outputs=gr.Textbox(label="Robot Action"),
        title="OpenVLA Robot Action Prediction",
        description="Provide an image of the robot workspace and an instruction to predict the robot's action. You can either upload an image or provide a base64-encoded image."
    )
    return gr_interface


# Launch the server on port 3389
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=3389)
