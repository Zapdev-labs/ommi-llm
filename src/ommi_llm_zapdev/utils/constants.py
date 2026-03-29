"""
Constants and configuration.
"""

# Supported architectures
SUPPORTED_ARCHITECTURES = [
    # Llama family
    "LlamaForCausalLM",
    "LlamaModel",
    # Mistral family
    "MistralForCausalLM",
    "MistralModel",
    # Mixtral MoE
    "MixtralForCausalLM",
    "MixtralModel",
    # Qwen family
    "Qwen2ForCausalLM",
    "QwenForCausalLM",
    "Qwen2Model",
    "QwenModel",
    # Baichuan
    "BaichuanForCausalLM",
    "BaichuanModel",
    # ChatGLM
    "ChatGLMForConditionalGeneration",
    "ChatGLMModel",
    # InternLM
    "InternLMForCausalLM",
    "InternLM2ForCausalLM",
    "InternLMModel",
]

# Default configurations
DEFAULT_MAX_MEMORY = {0: "max"}  # Use max available VRAM on GPU 0
DEFAULT_DTYPE = "float16"
DEFAULT_DEVICE = "cuda"
DEFAULT_PREFETCHING = True

# Compression options
COMPRESSION_OPTIONS = [None, "4bit", "8bit"]

# Model size estimates (in billions of parameters)
MODEL_SIZE_ESTIMATES = {
    "7b": 7,
    "13b": 13,
    "30b": 30,
    "34b": 34,
    "70b": 70,
    "72b": 72,
    "405b": 405,
}
