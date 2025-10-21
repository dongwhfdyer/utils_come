#!/usr/bin/env python3
"""
Example: Using the new LLM Client Manager

This script demonstrates how to use the refactored LLM client system.
"""

from llm_client import LLMClientManager, list_available_models, generate_text


def example_1_list_models():
    """Example 1: List all available models"""
    print("=" * 80)
    print("Example 1: List Available Models")
    print("=" * 80)

    models = list_available_models()
    print(f"\n{len(models)} models configured:")
    for i, model_id in enumerate(models, 1):
        print(f"  {i}. {model_id}")

    print()


def example_2_basic_chat():
    """Example 2: Basic chat with default model"""
    print("=" * 80)
    print("Example 2: Basic Chat (Default Model)")
    print("=" * 80)

    manager = LLMClientManager()

    # Get default model info
    model_config = manager.get_model_info()
    print(f"\nUsing: {model_config.model_id} ({model_config.description})")

    # Simple chat
    response = manager.chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in one sentence."}
        ],
        max_tokens=50
    )

    print(f"\nResponse: {response}\n")


def example_3_specific_model():
    """Example 3: Use a specific model"""
    print("=" * 80)
    print("Example 3: Using Specific Model")
    print("=" * 80)

    # Use convenience function
    response = generate_text(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2? Answer in one sentence."}
        ],
        model_id="qwen3-32b-dashscope",  # Explicitly specify model
        max_tokens=30
    )

    print(f"\nModel: qwen3-32b-dashscope")
    print(f"Response: {response}\n")


def example_4_caption_generation():
    """Example 4: Generate audio caption"""
    print("=" * 80)
    print("Example 4: Caption Generation")
    print("=" * 80)

    manager = LLMClientManager()

    system_prompt = "You are an audio analysis expert. Generate technical captions."

    user_prompt = """Based on these audio features, generate a 30-word technical caption:
- Spectral centroid: mel bin 25.3
- Dominant bin: 18
- Temporal std: 12.5 dB
- Stationarity: 0.234
- Peaks: 3
"""

    caption = manager.generate_caption(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=100,
        temperature=0.5
    )

    print(f"\nGenerated caption:\n{caption}\n")


def example_5_compare_models():
    """Example 5: Compare responses from different models"""
    print("=" * 80)
    print("Example 5: Compare Multiple Models")
    print("=" * 80)

    manager = LLMClientManager()

    prompt_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Describe a sunset in 15 words."}
    ]

    # Test with first 3 available models
    available_models = manager.list_models()[:3]

    print(f"\nTesting {len(available_models)} models:\n")

    for model_id in available_models:
        try:
            model_config = manager.get_model_info(model_id)
            print(f"Model: {model_id}")
            print(f"  Provider: {model_config.provider}")

            response = manager.chat_completion(
                messages=prompt_messages,
                model_id=model_id,
                max_tokens=50,
                temperature=0.7
            )

            print(f"  Response: {response}")
            print()

        except Exception as e:
            print(f"  Error: {e}")
            print(f"  (This is normal if API key not configured)\n")


def example_6_direct_client():
    """Example 6: Get direct access to OpenAI client"""
    print("=" * 80)
    print("Example 6: Direct Client Access")
    print("=" * 80)

    manager = LLMClientManager()

    # Get client and config
    client, model_config = manager.create_client("qwen3-32b-dashscope")

    print(f"\nModel ID: {model_config.model_id}")
    print(f"Provider: {model_config.provider}")
    print(f"API Base: {model_config.api_base}")
    print(f"Model Name: {model_config.model_name}")

    # Use client directly
    response = client.chat.completions.create(
        model=model_config.model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Count to 3."}
        ],
        max_tokens=20
    )

    result = response.choices[0].message.content
    print(f"\nDirect client response: {result}\n")


def main():
    """Run all examples"""
    print("\n" + "=" * 80)
    print("LLM Client Manager Examples")
    print("=" * 80 + "\n")

    try:
        # Example 1: List models
        example_1_list_models()

        # Example 2: Basic chat
        example_2_basic_chat()

        # Example 3: Specific model
        # example_3_specific_model()

        # Example 4: Caption generation
        # example_4_caption_generation()

        # Example 5: Compare models (commented out by default)
        # example_5_compare_models()

        # Example 6: Direct client (commented out by default)
        # example_6_direct_client()

    except Exception as e:
        print(f"\nError: {e}")
        print("\nNote: Some examples require API keys to be configured in .env")
        print("Set the appropriate environment variables:")
        print("  - DASHSCOPE_API_KEY for DashScope models")
        print("  - SGLANG_API_KEY for SGLang models (or 'EMPTY' for local)")
        print("  - DOUABAO_API_KEY for Doubao models")
        print("  - OPENAI_API_KEY for OpenAI models")

    print("\n" + "=" * 80)
    print("Examples Complete")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
