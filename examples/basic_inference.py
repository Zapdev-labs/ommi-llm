"""
Basic inference example.
"""

from ommi_llm import AutoModel


def main():
    # Load a model (automatically detects architecture)
    print("Loading model...")
    model = AutoModel.from_pretrained(
        "garage-bAInd/Platypus2-70B-instruct", device="cuda", dtype="float16", prefetching=True
    )

    # Prepare input
    input_text = ["What is the capital of United States?"]

    print(f"Input: {input_text[0]}")

    # Tokenize
    input_tokens = model.tokenizer(
        input_text,
        return_tensors="pt",
        return_attention_mask=False,
        truncation=True,
        max_length=128,
        padding=True,
    )

    # Generate
    print("Generating...")
    output_ids = model.generate(
        input_tokens["input_ids"].to(model.device), max_new_tokens=50, temperature=0.7, top_p=0.9
    )

    # Decode
    output_text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"Output: {output_text}")


if __name__ == "__main__":
    main()
