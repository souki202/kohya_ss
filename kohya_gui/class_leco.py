import gradio as gr
from .common_gui import get_any_file_path, document_symbol


class lecoTraining:
    def __init__(
        self,
        headless: bool = False,
        config: dict = {},
    ) -> None:
        self.headless = headless
        self.config = config

        with gr.Accordion(
            "LECO",
            open=True,
            visible=False,
            elem_classes=["leco_background"],
        ) as self.leco_accordion:
            gr.Markdown(
                "LECO trains a LoRA using text prompts only (no image dataset). "
                "Specify a prompt TOML file defining concept pairs."
            )
            with gr.Group():
                with gr.Row():
                    self.prompts_file = gr.Textbox(
                        label="Prompts File",
                        placeholder="Path to LECO prompt TOML file (required)",
                        value=self.config.get("leco.prompts_file", ""),
                        interactive=True,
                    )
                    self.prompts_file_button = gr.Button(
                        document_symbol,
                        elem_id="open_folder_small",
                        visible=(not headless),
                        interactive=True,
                    )
                    self.prompts_file_button.click(
                        get_any_file_path,
                        outputs=self.prompts_file,
                        show_progress=False,
                    )
                with gr.Row():
                    self.max_denoising_steps = gr.Slider(
                        label="Max Denoising Steps",
                        value=self.config.get("leco.max_denoising_steps", 40),
                        minimum=1,
                        maximum=200,
                        step=1,
                        info="Number of partial denoising steps per iteration",
                        interactive=True,
                    )
                    self.leco_denoise_guidance_scale = gr.Slider(
                        label="LECO Denoise Guidance Scale",
                        value=self.config.get("leco.leco_denoise_guidance_scale", 3.0),
                        minimum=0.0,
                        maximum=30.0,
                        step=0.1,
                        info="Guidance scale for the partial denoising pass",
                        interactive=True,
                    )
