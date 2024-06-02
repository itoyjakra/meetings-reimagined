"""
This module defines a LLMConfig class for configuring different
language models for various platforms.
"""

import os

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_aws import ChatBedrock
from pydantic import BaseModel
import boto3


class LLMConfig(BaseModel):
    platform: str
    model: str
    temperature: float = 0.5
    base_url: str = None

    def get_llm_model(self):
        """Returns an instance of the language model based
        on the configured platform and model."""

        if self.platform.lower() == "openai":
            return ChatOpenAI(model_name=self.model, temperature=self.temperature)

        if self.platform.lower() == "groq":
            return ChatGroq(
                model=self.model,
                api_key=os.getenv("GROQ_API_KEY"),
                temperature=self.temperature,
            )

        if self.platform.lower() == "ollama":
            os.environ["OPENAI_API_KEY"] = "NA"
            return ChatOpenAI(model=self.model, base_url="http://localhost:11434/v1")

        if self.platform.lower() == "bedrock":
            bedrock_client = boto3.client("bedrock-runtime")
            return ChatBedrock(
                model_id=self.model,
                model_kwargs={"temperature": 0.6},
                client=bedrock_client,
            )


# # Example usage:
# llm_config = LLMConfig(platform="openai", model="gpt-3.5-turbo")
# llm_model = llm_config.get_llm_model()
