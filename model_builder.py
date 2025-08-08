from .sampler.chat_completion_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    OPENAI_SYSTEM_MESSAGE_CHATGPT,
    ChatCompletionSampler,
)
from .sampler.openrouter_chat_completion_sampler import (
    OPENROUTER_SYSTEM_MESSAGE_API,
    OpenRouterChatCompletionSampler
)
from .sampler.claude_sampler import ClaudeCompletionSampler, CLAUDE_SYSTEM_MESSAGE_LMSYS
from .sampler.o_chat_completion_sampler import OChatCompletionSampler
from .sampler.responses_sampler import ResponsesSampler

class ModelBuilder:
    """Builds model instances based on the selected model name."""
    SUPPORTED_MODELS = [
        "o3",
        "o3_high",
        "o3_low",
        "o3-temp-1",
        "o4-mini",
        "o4-mini_high",
        "o4-mini_low",
        "o1-pro",
        "o1",
        "o1_high",
        "o1_low",
        "o1-preview",
        "o1-mini",
        "o3-mini",
        "o3-mini_high",
        "o3-mini_low",
        "gpt-4.1",
        "gpt-4.1-temp-1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4o",
        "gpt-4o-2024-11-20",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-08-06-temp-1",
        "gpt-4o-2024-05-13",
        "gpt-4o-mini",
        "gpt-4.5-preview",
        "gpt-4-turbo-2024-04-09",
        "gpt-4-0613",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-0125-temp-1",
        "chatgpt-4o-latest",
        "gpt-4-turbo-2024-04-09_chatgpt",
        "claude-3-opus-20240229_empty",
        "claude-3-7-sonnet-20250219",
        "claude-3-haiku-20240307",
        "openai/gpt-oss-20b",
    ]

    def build(self, model_name):
        match model_name:
            # Reasoning Models
            case "o3":
                return ResponsesSampler(
                    model="o3-2025-04-16",
                    reasoning_model=True,
                )
            case "o3-temp-1":
                return ResponsesSampler(
                    model="o3-2025-04-16",
                    reasoning_model=True,
                    temperature=1.0,
                )
            case "o3_high":
                return ResponsesSampler(
                    model="o3-2025-04-16",
                    reasoning_model=True,
                    reasoning_effort="high",
                )
            case "o3_low":
                return ResponsesSampler(
                    model="o3-2025-04-16",
                    reasoning_model=True,
                    reasoning_effort="low",
                )
            case "o4-mini":
                return ResponsesSampler(
                    model="o4-mini-2025-04-16",
                    reasoning_model=True,
                )
            case "o4-mini_high":
                return ResponsesSampler(
                    model="o4-mini-2025-04-16",
                    reasoning_model=True,
                    reasoning_effort="high",
                )
            case "o4-mini_low":
                return ResponsesSampler(
                    model="o4-mini-2025-04-16",
                    reasoning_model=True,
                    reasoning_effort="low",
                )
            case "o1-pro":
                return ResponsesSampler(
                    model="o1-pro",
                    reasoning_model=True,
                )
            case "o1":
                return OChatCompletionSampler(
                    model="o1",
                )
            case "o1_high":
                return OChatCompletionSampler(
                    model="o1",
                    reasoning_effort="high",
                )
            case "o1_low":
                return OChatCompletionSampler(
                    model="o1",
                    reasoning_effort="low",
                )
            case "o1-preview":
                return OChatCompletionSampler(
                    model="o1-preview",
                )
            case "o1-mini":
                return OChatCompletionSampler(
                    model="o1-mini",
                )
            # Default == Medium
            case "o3-mini":
                return OChatCompletionSampler(
                    model="o3-mini",
                )
            case "o3-mini_high":
                return OChatCompletionSampler(
                    model="o3-mini",
                    reasoning_effort="high",
                )
            case "o3-mini_low":
                return OChatCompletionSampler(
                    model="o3-mini",
                    reasoning_effort="low",
                )
            # GPT-4.1 models
            case "gpt-4.1":
                return ChatCompletionSampler(
                    model="gpt-4.1-2025-04-14",
                    system_message=OPENAI_SYSTEM_MESSAGE_API,
                    max_tokens=2048,
                )
            case "gpt-4.1-temp-1":
                return ChatCompletionSampler(
                    model="gpt-4.1-2025-04-14",
                    system_message=OPENAI_SYSTEM_MESSAGE_API,
                    max_tokens=2048,
                    temperature=1.0,
                )
            case "gpt-4.1-mini":
                return ChatCompletionSampler(
                    model="gpt-4.1-mini-2025-04-14",
                    system_message=OPENAI_SYSTEM_MESSAGE_API,
                    max_tokens=2048,
                )
            case "gpt-4.1-nano":
                return ChatCompletionSampler(
                    model="gpt-4.1-nano-2025-04-14",
                    system_message=OPENAI_SYSTEM_MESSAGE_API,
                    max_tokens=2048,
                )
            # GPT-4o models
            case "gpt-4o":
                return ChatCompletionSampler(
                    model="gpt-4o",
                    system_message=OPENAI_SYSTEM_MESSAGE_API,
                    max_tokens=2048,
                )
            case "gpt-4o-2024-11-20":
                return ChatCompletionSampler(
                    model="gpt-4o-2024-11-20",
                    system_message=OPENAI_SYSTEM_MESSAGE_API,
                    max_tokens=2048,
                )
            case "gpt-4o-2024-08-06":
                return ChatCompletionSampler(
                    model="gpt-4o-2024-08-06",
                    system_message=OPENAI_SYSTEM_MESSAGE_API,
                    max_tokens=2048,
                )
            case "gpt-4o-2024-08-06-temp-1":
                return ChatCompletionSampler(
                    model="gpt-4o-2024-08-06",
                    system_message=OPENAI_SYSTEM_MESSAGE_API,
                    max_tokens=2048,
                    temperature=1.0,
                )
            case "gpt-4o-2024-05-13":
                return ChatCompletionSampler(
                    model="gpt-4o-2024-05-13",
                    system_message=OPENAI_SYSTEM_MESSAGE_API,
                    max_tokens=2048,
                )
            case "gpt-4o-mini":
                return ChatCompletionSampler(
                    model="gpt-4o-mini-2024-07-18",
                    system_message=OPENAI_SYSTEM_MESSAGE_API,
                    max_tokens=2048,
                )
            # GPT-4.5 model
            case "gpt-4.5-preview":
                return ChatCompletionSampler(
                    model="gpt-4.5-preview-2025-02-27",
                    system_message=OPENAI_SYSTEM_MESSAGE_API,
                    max_tokens=2048,
                )
            # GPT-4-turbo model
            case "gpt-4-turbo-2024-04-09":
                return ChatCompletionSampler(
                    model="gpt-4-turbo-2024-04-09",
                    system_message=OPENAI_SYSTEM_MESSAGE_API,
                )
            # GPT-4 model
            case "gpt-4-0613":
                return ChatCompletionSampler(
                    model="gpt-4-0613",
                    system_message=OPENAI_SYSTEM_MESSAGE_API,
                )
            # GPT-3.5 Turbo model
            case "gpt-3.5-turbo-0125":
                return ChatCompletionSampler(
                    model="gpt-3.5-turbo-0125",
                    system_message=OPENAI_SYSTEM_MESSAGE_API,
                )
            case "gpt-3.5-turbo-0125-temp-1":
                return ChatCompletionSampler(
                    model="gpt-3.5-turbo-0125",
                    system_message=OPENAI_SYSTEM_MESSAGE_API,
                    temperature=1.0,
                )
            # Chatgpt models:
            case "chatgpt-4o-latest":
                return ChatCompletionSampler(
                    model="chatgpt-4o-latest",
                    system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
                    max_tokens=2048,
                )
            case "gpt-4-turbo-2024-04-09_chatgpt":
                return ChatCompletionSampler(
                    model="gpt-4-turbo-2024-04-09",
                    system_message=OPENAI_SYSTEM_MESSAGE_CHATGPT,
                )
            # Claude models:
            case "claude-3-opus-20240229_empty":
                return ClaudeCompletionSampler(
                    model="claude-3-opus-20240229",
                    system_message=CLAUDE_SYSTEM_MESSAGE_LMSYS,
                )
            case "claude-3-7-sonnet-20250219":
                return ClaudeCompletionSampler(
                    model="claude-3-7-sonnet-20250219",
                    system_message=CLAUDE_SYSTEM_MESSAGE_LMSYS,
                )
            case "claude-3-haiku-20240307":
                return ClaudeCompletionSampler(
                    model="claude-3-haiku-20240307",
                )
            # OpenRouter models:
            case "openai/gpt-oss-20b":
                return OpenRouterChatCompletionSampler(
                    model="openai/gpt-oss-20b",
                    system_message=OPENROUTER_SYSTEM_MESSAGE_API,
                )
            case "openai/gpt-oss-200b":
                return OpenRouterChatCompletionSampler(
                    model="openai/gpt-oss-200b",
                    system_message=OPENROUTER_SYSTEM_MESSAGE_API,
                )
            case "qwen/qwen3-coder_chutes/fp8":
                return OpenRouterChatCompletionSampler(
                    model="qwen/qwen3-coder",
                    provider="chutes/fp8",
                )
            case "qwen/qwen3-coder_chutes":
                return OpenRouterChatCompletionSampler(
                    model="qwen/qwen3-coder",
                    provider="chutes",
                )
            case "qwen/qwen3-coder_targon/fp8":
                return OpenRouterChatCompletionSampler(
                    model="qwen/qwen3-coder",
                    provider="baseten/fp8",
                )
            case "qwen/qwen3-coder_targon/fp8":
                return OpenRouterChatCompletionSampler(
                    model="qwen/qwen3-coder",
                    provider="targon/fp8",
                )
            case "qwen/qwen3-coder_phala":
                return OpenRouterChatCompletionSampler(
                    model="qwen/qwen3-coder",
                    provider="phala",
                )
            case "qwen/qwen3-coder_together/fp8":
                return OpenRouterChatCompletionSampler(
                    model="qwen/qwen3-coder",
                    provider="together/fp8",
                )
            case "qwen/qwen3-coder_hyperbolic/fp8":
                return OpenRouterChatCompletionSampler(
                    model="qwen/qwen3-coder",
                    provider="hyperbolic/fp8",
                )
            case _:
                raise Exception(f"Unsupported model name: {model_name}")