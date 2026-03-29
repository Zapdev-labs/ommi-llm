"""
Model sharding example.
"""

from pathlib import Path
from ommi_llm.persistence.sharder import ModelSharder


def main():
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    output_path = Path("./sharded-models/llama-2-7b")

    print(f"Sharding model: {model_name}")
    print(f"Output: {output_path}")

    # Create sharder with 4-bit compression
    sharder = ModelSharder(model_name, output_path, compression="4bit")

    # Shard the model
    shard_paths = sharder.shard_model(delete_original=False, skip_if_exists=True)

    print(f"\nModel sharded into {len(shard_paths)} layers:")
    for path in shard_paths[:5]:  # Show first 5
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  - {path.name}: {size_mb:.2f} MB")

    if len(shard_paths) > 5:
        print(f"  ... and {len(shard_paths) - 5} more layers")


if __name__ == "__main__":
    main()
