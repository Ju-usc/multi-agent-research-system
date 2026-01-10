import os

from dotenv import load_dotenv

load_dotenv(override=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PARALLEL_API_KEY = os.getenv("PARALLEL_API_KEY")

DEFAULT_LEAD_MODEL = "openrouter/x-ai/grok-4.1-fast"
DEFAULT_SUB_MODEL = "openrouter/x-ai/grok-4.1-fast"
DEFAULT_LEAD_MAX_TOKENS = 40000
DEFAULT_SUB_MAX_TOKENS = 40000
DEFAULT_TEMPERATURE = 1.0


class ModelConfig:
    def __init__(
        self,
        lead: str = DEFAULT_LEAD_MODEL,
        sub: str = DEFAULT_SUB_MODEL,
        lead_max_tokens: int = DEFAULT_LEAD_MAX_TOKENS,
        sub_max_tokens: int = DEFAULT_SUB_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        offline: bool = False,
    ):
        self.lead = lead
        self.sub = sub
        self.lead_max_tokens = lead_max_tokens
        self.sub_max_tokens = sub_max_tokens
        self.temperature = temperature
        self.offline = offline


GRADER_MODEL = "openrouter/openai/gpt-5.2"
GRADER_MAX_TOKENS = 40000
OPTIMIZER_MODEL = "openrouter/openai/gpt-5.2"
OPTIMIZER_MAX_TOKENS = 40000
