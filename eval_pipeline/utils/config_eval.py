import os
from dataclasses import dataclass
from typing import ClassVar, Dict

from dotenv import load_dotenv
from fastchat.model import get_conversation_template

load_dotenv()

BASE_PATH = os.getenv('BASE_PATH')

@dataclass
class ConvTemplates:

    template_names: ClassVar[Dict[str, str]] = {
        "llama31-8b": "llama-3",
        "gemma2-9b": "gemma",
        "phi4-14b": "phi-4"
    }

    def get_template_name(self, model_name: str) -> str:
        return self.template_names.get(model_name, "Template not found")

    def get_template(self, model_name: str):
        template = get_conversation_template(self.get_template_name(model_name))
        return template
