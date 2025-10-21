# LLM Client Configuration System

This directory contains a refactored LLM calling module with centralized configuration management.

## Overview

The new system provides:
- **Centralized Configuration**: All model configurations in one YAML file
- **Provider Support**: Support for multiple providers (DashScope, SGLang, Doubao, OpenAI, etc.)
- **Easy Model Switching**: Switch between models using simple identifiers
- **OpenAI-Compatible**: All models use OpenAI-compatible APIs

## Files

- `model_config.yaml` - Model configuration file
- `llm_client.py` - Refactored LLM client manager
- `generate_caption.py` - Legacy caption generator (maintained for backward compatibility)

## Model Configuration (`model_config.yaml`)

Each model is defined with:
- `provider`: Provider name (dashscope, sglang, doubao, openai, etc.)
- `api_base`: API base URL
- `api_key_env`: Environment variable name for API key
- `model_name`: Actual model name to pass to the API
- `description`: Human-readable description
- `max_tokens`: Default max tokens
- `temperature`: Default temperature

### Example Model Configuration

```yaml
models:
  qwen3-32b-dashscope:
    provider: "dashscope"
    api_base: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_env: "DASHSCOPE_API_KEY"
    model_name: "qwen-plus"
    description: "Qwen3-32B via DashScope API"
    max_tokens: 4096
    temperature: 0.7
```

## Usage

### 1. Basic Usage

```python
from llm_client import LLMClientManager

# Initialize manager (reads from model_config.yaml)
manager = LLMClientManager()

# List available models
print(manager.list_models())

# Generate text using default model
response = manager.chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
)
```

### 2. Specify Model

```python
# Use a specific model
response = manager.chat_completion(
    messages=[...],
    model_id="qwen3-32b-dashscope"
)

# Or use local model
response = manager.chat_completion(
    messages=[...],
    model_id="qwen3-32b-local-sglang"
)
```

### 3. Caption Generation

```python
# Convenience method for caption generation
caption = manager.generate_caption(
    system_prompt="You are an audio expert.",
    user_prompt="Describe this audio...",
    model_id="qwen3-32b-dashscope",
    max_tokens=150,
    temperature=0.5
)
```

### 4. Get Client Directly

```python
# Get OpenAI client and config
client, model_config = manager.create_client("qwen3-32b-dashscope")

# Use client directly
response = client.chat.completions.create(
    model=model_config.model_name,
    messages=[...]
)
```

### 5. Convenience Functions

```python
from llm_client import list_available_models, generate_text

# List models
models = list_available_models()

# Generate text
response = generate_text(
    messages=[...],
    model_id="qwen3-32b-dashscope"
)
```

## Environment Variables

Set API keys in your `.env` file:

```bash
# DashScope (Alibaba Cloud)
DASHSCOPE_API_KEY=your_dashscope_key

# SGLang (local models, can use dummy key)
SGLANG_API_KEY=EMPTY

# Doubao
DOUABAO_API_KEY=your_doubao_key

# OpenAI
OPENAI_API_KEY=your_openai_key
```

## Adding New Models

To add a new model, edit `model_config.yaml`:

```yaml
models:
  my-custom-model:
    provider: "custom"
    api_base: "https://my-api.com/v1"
    api_key_env: "MY_API_KEY"
    model_name: "my-model-name"
    description: "My custom model"
    max_tokens: 4096
    temperature: 0.7
```

Then set the API key in `.env`:

```bash
MY_API_KEY=your_key_here
```

## Updated Scripts

The following scripts have been updated to use the new LLM client:

- `src/validators/test_single_sample.py`
- `src/validators/generate_10_samples.py`

### Example: Using in CaptionStyleGenerator

```python
from llm_client import LLMClientManager

class CaptionStyleGenerator:
    def __init__(self, style: str = "hybrid", model_id: str = None):
        self.style = style
        self.model_id = model_id
        self.llm_manager = LLMClientManager()

    def generate_caption(self, features):
        caption = self.llm_manager.generate_caption(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model_id=self.model_id,
            max_tokens=150,
            temperature=0.5
        )
        return caption
```

## Model Naming Convention

Model identifiers follow the pattern: `{model-name}-{provider}`

Examples:
- `qwen3-32b-dashscope` - Qwen3-32B via DashScope
- `qwen3-32b-local-sglang` - Qwen3-32B via local SGLang
- `qwen2-7b-local-sglang` - Qwen2-7B via local SGLang
- `doubao-pro-32k` - Doubao Pro 32K
- `gpt-4o-openai` - GPT-4o via OpenAI

## Testing

Test the LLM client:

```bash
# Test the client manager
python src/core/llm_client.py

# Test with validators
python src/validators/test_single_sample.py
```

## Backward Compatibility

The original `generate_caption.py` is maintained for backward compatibility with existing scripts that use `AudioCaptionGenerator`. New code should use `llm_client.py` for better flexibility and centralized configuration.

## Default Model

The default model is configured in `model_config.yaml`:

```yaml
default_model: "qwen3-32b-dashscope"
```

This model will be used when no `model_id` is specified.

## Timeout Configuration

Timeouts can be configured globally in `model_config.yaml`:

```yaml
timeout:
  connect: 10
  read: 60
  write: 10
```

## Benefits

1. **Centralized Configuration**: All models in one place
2. **Easy Switching**: Change models without code changes
3. **Provider Agnostic**: Works with any OpenAI-compatible API
4. **Type Safe**: Clear configuration schema
5. **Environment-Based**: API keys stored securely in .env
6. **Flexible**: Supports multiple models from multiple providers
7. **Testable**: Easy to test different models
