import gradio as gr
from .common_gui import get_any_file_path, document_symbol


class luminaTraining:
    def __init__(
        self,
        headless: bool = False,
        finetuning: bool = False,
        training_type: str = "",
        config: dict = {},
        lumina_checkbox: gr.Checkbox = False,
    ) -> None:
        self.headless = headless
        self.finetuning = finetuning
        self.training_type = training_type
        self.config = config
        self.lumina_checkbox = lumina_checkbox

        with gr.Accordion(
            "Lumina",
            open=False,
            visible=False,
            elem_classes=["lumina_background"],
        ) as lumina_accordion:
            with gr.Group():
                with gr.Row():
                    self.gemma2 = gr.Textbox(
                        label="Gemma2 Path",
                        placeholder="Path to Gemma2 model",
                        value=self.config.get("lumina.gemma2", ""),
                        interactive=True,
                    )
                    self.gemma2_button = gr.Button(
                        document_symbol,
                        elem_id="open_folder_small",
                        visible=(not headless),
                        interactive=True,
                    )
                    self.gemma2_button.click(
                        get_any_file_path,
                        outputs=self.gemma2,
                        show_progress=False,
                    )

                    self.ae = gr.Textbox(
                        label="Autoencoder Path",
                        placeholder="Path to autoencoder model",
                        value=self.config.get("lumina.ae", ""),
                        interactive=True,
                    )
                    self.ae_button = gr.Button(
                        document_symbol,
                        elem_id="open_folder_small",
                        visible=(not headless),
                        interactive=True,
                    )
                    self.ae_button.click(
                        get_any_file_path,
                        outputs=self.ae,
                        show_progress=False,
                    )

                with gr.Row():
                    self.use_flash_attn = gr.Checkbox(
                        label="Use Flash Attention",
                        value=self.config.get("lumina.use_flash_attn", False),
                        interactive=True,
                    )
                    self.use_sage_attn = gr.Checkbox(
                        label="Use Sage Attention",
                        value=self.config.get("lumina.use_sage_attn", False),
                        interactive=True,
                    )
                    self.sample_batch_size = gr.Number(
                        label="Sample Batch Size",
                        value=self.config.get("lumina.sample_batch_size", 0),
                        info="Optional batch size for validation sampling",
                        minimum=0,
                        maximum=4096,
                        step=1,
                        interactive=True,
                    )

                with gr.Row():
                    self.system_prompt = gr.Textbox(
                        label="System Prompt",
                        placeholder="Optional system prompt",
                        value=self.config.get("lumina.system_prompt", ""),
                        interactive=True,
                    )

                with gr.Row():
                    self.gemma2_max_token_length = gr.Number(
                        label="Gemma2 Max Token Length",
                        value=self.config.get("lumina.gemma2_max_token_length", 256),
                        minimum=0,
                        maximum=4096,
                        step=1,
                        interactive=True,
                    )
                    self.timestep_sampling = gr.Dropdown(
                        label="Timestep Sampling",
                        choices=["sigma", "uniform", "sigmoid", "shift", "nextdit_shift"],
                        value=self.config.get("lumina.timestep_sampling", "shift"),
                        interactive=True,
                    )
                    self.sigmoid_scale = gr.Number(
                        label="Sigmoid Scale",
                        value=self.config.get("lumina.sigmoid_scale", 1.0),
                        minimum=0.0,
                        maximum=100.0,
                        step=0.1,
                        interactive=True,
                    )

                with gr.Row():
                    self.model_prediction_type = gr.Dropdown(
                        label="Model Prediction Type",
                        choices=["raw", "additive", "sigma_scaled"],
                        value=self.config.get("lumina.model_prediction_type", "raw"),
                        interactive=True,
                    )
                    self.discrete_flow_shift = gr.Number(
                        label="Discrete Flow Shift",
                        value=self.config.get("lumina.discrete_flow_shift", 6.0),
                        minimum=-1024.0,
                        maximum=1024.0,
                        step=0.1,
                        interactive=True,
                    )

            if not finetuning:
                with gr.Group():
                    with gr.Row():
                        self.train_blocks = gr.Dropdown(
                            label="Train Blocks",
                            choices=["all", "transformer", "refiners", "noise_refiner", "context_refiner"],
                            value=self.config.get("lumina.train_blocks", "all"),
                            interactive=True,
                        )
                        self.split_qkv = gr.Checkbox(
                            label="Split QKV",
                            value=self.config.get("lumina.split_qkv", False),
                            interactive=True,
                        )
                        self.verbose = gr.Checkbox(
                            label="Verbose Network Logs",
                            value=self.config.get("lumina.verbose", False),
                            interactive=True,
                        )
                    with gr.Row():
                        self.attn_dim = gr.Textbox(
                            label="Attention Rank",
                            placeholder="Optional rank for attention layers",
                            value=self.config.get("lumina.attn_dim", ""),
                            interactive=True,
                        )
                        self.mlp_dim = gr.Textbox(
                            label="MLP Rank",
                            placeholder="Optional rank for MLP layers",
                            value=self.config.get("lumina.mlp_dim", ""),
                            interactive=True,
                        )
                        self.mod_dim = gr.Textbox(
                            label="Modulation Rank",
                            placeholder="Optional rank for modulation layers",
                            value=self.config.get("lumina.mod_dim", ""),
                            interactive=True,
                        )
                        self.refiner_dim = gr.Textbox(
                            label="Refiner Rank",
                            placeholder="Optional rank for refiner blocks",
                            value=self.config.get("lumina.refiner_dim", ""),
                            interactive=True,
                        )
                    with gr.Row():
                        self.embedder_dims = gr.Textbox(
                            label="Embedder Ranks",
                            placeholder="Optional list for embedder ranks, e.g. [4,0,0]",
                            value=self.config.get("lumina.embedder_dims", ""),
                            interactive=True,
                        )
            else:
                self.train_blocks = gr.Textbox(value="", visible=False)
                self.split_qkv = gr.Checkbox(value=False, visible=False)
                self.verbose = gr.Checkbox(value=False, visible=False)
                self.attn_dim = gr.Textbox(value="", visible=False)
                self.mlp_dim = gr.Textbox(value="", visible=False)
                self.mod_dim = gr.Textbox(value="", visible=False)
                self.refiner_dim = gr.Textbox(value="", visible=False)
                self.embedder_dims = gr.Textbox(value="", visible=False)

        self.lumina_checkbox.change(
            lambda lumina_checkbox: gr.Accordion(visible=lumina_checkbox),
            inputs=[self.lumina_checkbox],
            outputs=[lumina_accordion],
        )
