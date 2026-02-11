import gradio as gr
from .common_gui import get_any_file_path, document_symbol


class animaTraining:
    def __init__(
        self,
        headless: bool = False,
        finetuning: bool = False,
        training_type: str = "",
        config: dict = {},
        anima_checkbox: gr.Checkbox = False,
    ) -> None:
        self.headless = headless
        self.finetuning = finetuning
        self.training_type = training_type
        self.config = config
        self.anima_checkbox = anima_checkbox

        with gr.Accordion(
            "Anima",
            open=False,
            visible=False,
            elem_classes=["anima_background"],
        ) as anima_accordion:
            with gr.Group():
                with gr.Row():
                    self.dit_path = gr.Textbox(
                        label="DiT Path",
                        placeholder="Path to Anima DiT model safetensors file",
                        value=self.config.get("anima.dit_path", ""),
                        interactive=True,
                    )
                    self.dit_path_button = gr.Button(
                        document_symbol,
                        elem_id="open_folder_small",
                        visible=(not headless),
                        interactive=True,
                    )
                    self.dit_path_button.click(
                        get_any_file_path,
                        outputs=self.dit_path,
                        show_progress=False,
                    )

                    self.vae_path = gr.Textbox(
                        label="VAE Path",
                        placeholder="Path to WanVAE safetensors/pth file",
                        value=self.config.get("anima.vae_path", ""),
                        interactive=True,
                    )
                    self.vae_path_button = gr.Button(
                        document_symbol,
                        elem_id="open_folder_small",
                        visible=(not headless),
                        interactive=True,
                    )
                    self.vae_path_button.click(
                        get_any_file_path,
                        outputs=self.vae_path,
                        show_progress=False,
                    )

                with gr.Row():
                    self.qwen3_path = gr.Textbox(
                        label="Qwen3 Path",
                        placeholder="Path to Qwen3-0.6B model (safetensors file or directory)",
                        value=self.config.get("anima.qwen3_path", ""),
                        interactive=True,
                    )
                    self.qwen3_path_button = gr.Button(
                        document_symbol,
                        elem_id="open_folder_small",
                        visible=(not headless),
                        interactive=True,
                    )
                    self.qwen3_path_button.click(
                        get_any_file_path,
                        outputs=self.qwen3_path,
                        show_progress=False,
                    )

                    self.llm_adapter_path = gr.Textbox(
                        label="LLM Adapter Path",
                        placeholder="(Optional) Path to separate LLM adapter weights",
                        value=self.config.get("anima.llm_adapter_path", ""),
                        interactive=True,
                    )
                    self.llm_adapter_path_button = gr.Button(
                        document_symbol,
                        elem_id="open_folder_small",
                        visible=(not headless),
                        interactive=True,
                    )
                    self.llm_adapter_path_button.click(
                        get_any_file_path,
                        outputs=self.llm_adapter_path,
                        show_progress=False,
                    )

                with gr.Row():
                    self.t5_tokenizer_path = gr.Textbox(
                        label="T5 Tokenizer Path",
                        placeholder="(Optional) Path to T5 tokenizer directory",
                        value=self.config.get("anima.t5_tokenizer_path", ""),
                        interactive=True,
                    )
                    self.t5_tokenizer_path_button = gr.Button(
                        document_symbol,
                        elem_id="open_folder_small",
                        visible=(not headless),
                        interactive=True,
                    )
                    self.t5_tokenizer_path_button.click(
                        get_any_file_path,
                        outputs=self.t5_tokenizer_path,
                        show_progress=False,
                    )

                with gr.Row():
                    self.qwen3_max_token_length = gr.Number(
                        label="Qwen3 Max Token Length",
                        value=self.config.get("anima.qwen3_max_token_length", 512),
                        minimum=0,
                        maximum=4096,
                        step=1,
                        interactive=True,
                    )
                    self.t5_max_token_length = gr.Number(
                        label="T5 Max Token Length",
                        value=self.config.get("anima.t5_max_token_length", 512),
                        minimum=0,
                        maximum=4096,
                        step=1,
                        interactive=True,
                    )

                with gr.Row():
                    self.discrete_flow_shift = gr.Number(
                        label="Discrete Flow Shift",
                        value=self.config.get("anima.discrete_flow_shift", 1.0),
                        minimum=-1024.0,
                        maximum=1024.0,
                        step=0.1,
                        interactive=True,
                    )
                    self.timestep_sample_method = gr.Dropdown(
                        label="Timestep Sample Method",
                        choices=["logit_normal", "uniform"],
                        value=self.config.get("anima.timestep_sample_method", "logit_normal"),
                        interactive=True,
                    )
                    self.sigmoid_scale = gr.Number(
                        label="Sigmoid Scale",
                        value=self.config.get("anima.sigmoid_scale", 1.0),
                        minimum=0.0,
                        maximum=100.0,
                        step=0.1,
                        interactive=True,
                    )

                with gr.Row():
                    self.transformer_dtype = gr.Dropdown(
                        label="Transformer Dtype",
                        choices=["None", "float16", "bfloat16", "float32"],
                        value=self.config.get("anima.transformer_dtype", "None"),
                        interactive=True,
                    )
                    self.flash_attn = gr.Checkbox(
                        label="Flash Attention",
                        value=self.config.get("anima.flash_attn", False),
                        interactive=True,
                    )
                    self.cpu_offload_checkpointing = gr.Checkbox(
                        label="CPU Offload Checkpointing",
                        value=self.config.get("anima.cpu_offload_checkpointing", False),
                        interactive=True,
                    )
                    self.unsloth_offload_checkpointing = gr.Checkbox(
                        label="Unsloth Offload Checkpointing",
                        value=self.config.get("anima.unsloth_offload_checkpointing", False),
                        interactive=True,
                    )

            # Finetune-only: component-wise learning rates and blockwise fused optimizers
            if finetuning:
                with gr.Group():
                    with gr.Row():
                        self.blockwise_fused_optimizers = gr.Checkbox(
                            label="Blockwise Fused Optimizers",
                            value=self.config.get("anima.blockwise_fused_optimizers", False),
                            interactive=True,
                        )
                    with gr.Row():
                        self.llm_adapter_lr = gr.Textbox(
                            label="LLM Adapter LR",
                            placeholder="(Optional) LR for LLM adapter. 0=freeze",
                            value=self.config.get("anima.llm_adapter_lr", ""),
                            interactive=True,
                        )
                        self.self_attn_lr = gr.Textbox(
                            label="Self Attention LR",
                            placeholder="(Optional) LR for self-attention layers",
                            value=self.config.get("anima.self_attn_lr", ""),
                            interactive=True,
                        )
                        self.cross_attn_lr = gr.Textbox(
                            label="Cross Attention LR",
                            placeholder="(Optional) LR for cross-attention layers",
                            value=self.config.get("anima.cross_attn_lr", ""),
                            interactive=True,
                        )
                    with gr.Row():
                        self.mlp_lr = gr.Textbox(
                            label="MLP LR",
                            placeholder="(Optional) LR for MLP layers",
                            value=self.config.get("anima.mlp_lr", ""),
                            interactive=True,
                        )
                        self.mod_lr = gr.Textbox(
                            label="Modulation LR",
                            placeholder="(Optional) LR for AdaLN modulation layers",
                            value=self.config.get("anima.mod_lr", ""),
                            interactive=True,
                        )
            else:
                # Hidden placeholders when not finetuning
                self.blockwise_fused_optimizers = gr.Checkbox(value=False, visible=False)
                self.llm_adapter_lr = gr.Textbox(value="", visible=False)
                self.self_attn_lr = gr.Textbox(value="", visible=False)
                self.cross_attn_lr = gr.Textbox(value="", visible=False)
                self.mlp_lr = gr.Textbox(value="", visible=False)
                self.mod_lr = gr.Textbox(value="", visible=False)

            # LoRA-only: network args for per-component rank dims
            if not finetuning:
                with gr.Group():
                    with gr.Row():
                        self.self_attn_dim = gr.Textbox(
                            label="Self Attention Rank",
                            placeholder="Optional rank for self-attention LoRA layers",
                            value=self.config.get("anima.self_attn_dim", ""),
                            interactive=True,
                        )
                        self.cross_attn_dim = gr.Textbox(
                            label="Cross Attention Rank",
                            placeholder="Optional rank for cross-attention LoRA layers",
                            value=self.config.get("anima.cross_attn_dim", ""),
                            interactive=True,
                        )
                        self.mlp_dim = gr.Textbox(
                            label="MLP Rank",
                            placeholder="Optional rank for MLP LoRA layers",
                            value=self.config.get("anima.mlp_dim", ""),
                            interactive=True,
                        )
                    with gr.Row():
                        self.mod_dim = gr.Textbox(
                            label="Modulation Rank",
                            placeholder="Optional rank for modulation LoRA layers",
                            value=self.config.get("anima.mod_dim", ""),
                            interactive=True,
                        )
                        self.llm_adapter_dim = gr.Textbox(
                            label="LLM Adapter Rank",
                            placeholder="Optional rank for LLM adapter LoRA layers",
                            value=self.config.get("anima.llm_adapter_dim", ""),
                            interactive=True,
                        )
                        self.emb_dims = gr.Textbox(
                            label="Embedder Ranks",
                            placeholder="Optional list for embedder ranks, e.g. [4,0,0]",
                            value=self.config.get("anima.emb_dims", ""),
                            interactive=True,
                        )
                    with gr.Row():
                        self.train_block_indices = gr.Textbox(
                            label="Train Block Indices",
                            placeholder='Block selection: "all", "none", "0-5", "0,3,7"',
                            value=self.config.get("anima.train_block_indices", ""),
                            interactive=True,
                        )
                        self.train_llm_adapter = gr.Checkbox(
                            label="Train LLM Adapter",
                            value=self.config.get("anima.train_llm_adapter", False),
                            interactive=True,
                        )
                        self.verbose = gr.Checkbox(
                            label="Verbose Network Logs",
                            value=self.config.get("anima.verbose", False),
                            interactive=True,
                        )
            else:
                # Hidden placeholders when finetuning
                self.self_attn_dim = gr.Textbox(value="", visible=False)
                self.cross_attn_dim = gr.Textbox(value="", visible=False)
                self.mlp_dim = gr.Textbox(value="", visible=False)
                self.mod_dim = gr.Textbox(value="", visible=False)
                self.llm_adapter_dim = gr.Textbox(value="", visible=False)
                self.emb_dims = gr.Textbox(value="", visible=False)
                self.train_block_indices = gr.Textbox(value="", visible=False)
                self.train_llm_adapter = gr.Checkbox(value=False, visible=False)
                self.verbose = gr.Checkbox(value=False, visible=False)

        self.anima_checkbox.change(
            lambda anima_checkbox: gr.Accordion(visible=anima_checkbox),
            inputs=[self.anima_checkbox],
            outputs=[anima_accordion],
        )
