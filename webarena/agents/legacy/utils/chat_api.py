import json
from dataclasses import asdict, dataclass

@dataclass
class ChatModelArgs:
    model_name: str = "openai/gpt-4o"
    model_url: str = None
    temperature: float = 0.1
    max_new_tokens: int = None
    max_total_tokens: int = None
    max_input_tokens: int = None
    hf_hosted: bool = False
    info: dict = None
    n_retry_server: int = 4

    def has_vision(self):
        name_patterns_with_vision = ["vision", "4o"]
        return any(pattern in self.model_name for pattern in name_patterns_with_vision)
