"""
Using skills example.
"""

from ommi_llm import AutoModel
from ommi_llm.skills.registry import get_skill_registry


def main():
    # Load model
    print("Loading model with skills...")
    model = AutoModel.from_pretrained("mistralai/Mistral-7B-v0.1", device="cuda", dtype="float16")

    # Get skill registry
    registry = get_skill_registry()

    # List available skills
    print("\nAvailable skills:")
    for skill_info in registry.list_skills():
        print(f"  - {skill_info['name']}: {skill_info['description']}")

    # Apply skills
    print("\nApplying skills...")

    # Enable quantization
    registry.apply_skill("quantization", model, {"mode": "4bit"})

    # Configure KV cache
    registry.apply_skill("kv_cache", model, {"max_cache_size": 4096, "offload_to_cpu": False})

    # Enable Flash Attention (if available)
    registry.apply_skill("flash_attention", model, {"enabled": True})

    # Now use the model with optimizations applied
    print("\nModel ready with optimizations applied!")

    # Show model info
    info = model.get_model_info()
    print(f"\nModel info:")
    for key, value in info.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
