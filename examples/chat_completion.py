"""
Chat completion example.
"""

from ommi_llm import AutoModel


def chat_completion(model, messages, max_tokens=100):
    """Simple chat completion with message history."""
    # Format messages into a prompt
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt += f"System: {content}\n"
        elif role == "user":
            prompt += f"User: {content}\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n"

    prompt += "Assistant:"

    # Tokenize
    input_tokens = model.tokenizer(
        prompt, return_tensors="pt", return_attention_mask=False, truncation=True, max_length=4096
    )

    # Generate
    output_ids = model.generate(
        input_tokens["input_ids"].to(model.device),
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_p=0.9,
    )

    # Decode only the new tokens
    new_tokens = output_ids[0][input_tokens["input_ids"].shape[1] :]
    response = model.tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response.strip()


def main():
    print("Loading chat model...")
    model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device="cuda")

    # Chat history
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"},
    ]

    print("\nUser: What is machine learning?")

    # Get response
    response = chat_completion(model, messages)

    print(f"Assistant: {response}")

    # Continue conversation
    messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": "Can you give me a simple example?"})

    print("\nUser: Can you give me a simple example?")

    response = chat_completion(model, messages)

    print(f"Assistant: {response}")


if __name__ == "__main__":
    main()
