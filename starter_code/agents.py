import os
import re

from crewai import Agent
from crewai_tools import FileReadTool
from langchain.llms import Ollama, OpenAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from loguru import logger

transcription_reader = FileReadTool(file_path="./meetings/topic1/2024-05-11.txt")


class MeetingAgents:
    def __init__(
        self, llm_model, model_str: str = "gpt-3.5-turbo", temperature: float = 0.5
    ):
        self.llm_model = llm_model
        self.OpenAIGPT35 = ChatOpenAI(
            model_name="gpt-3.5-turbo", temperature=temperature
        )
        self.OpenAIGPT4 = ChatOpenAI(model_name="gpt-4o", temperature=temperature)
        self.Ollama = Ollama(model="openhermes")
        self.GROQ_LLM = ChatGroq(
            model="llama3-70b-8192",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=temperature,
        )
        self.OLLAMA_LLM = ChatOpenAI(
            model="crewai-llama3:80b", base_url="http://localhost:11434/v1"
        )
        self.model_str = model_str

    @property
    def llm(self):
        if re.search("openai", self.model_str, re.IGNORECASE):
            logger.debug(f"Using OpenAI model: {self.model_str}")
            if "gpt-3.5-turbo" in self.model_str:
                logger.debug(f"Using OpenAI GPT-3.5-turbo")
                return self.OpenAIGPT35
            elif "gpt-4" in self.model_str:
                logger.debug(f"Using OpenAI GPT-4")
                return self.OpenAIGPT4

        if "groq" in self.model_str and "llama3" in self.model_str:
            logger.debug(f"Using GROQ model: {self.model_str}")
            return self.GROQ_LLM

        if "ollama" in self.model_str and "llama3" in self.model_str:
            logger.debug(f"Using OLLAMA model: {self.model_str}")
            os.environ["OPENAI_API_KEY"] = "NA"
            return self.OLLAMA_LLM

    def summarizer(self):
        return Agent(
            role="Document Summarizer",
            backstory=(
                "You have participated in many technical meetings. "
                "You can identify the topic of the discussion and "
                "create a summary of the meeting capturing the "
                "essential topics. "
            ),
            goal="Create a summary of the meeting without any individual's name.",
            tools=[transcription_reader],
            allow_delegation=False,
            verbose=True,
            # llm=self.OpenAIGPT4,
            llm=self.llm_model,
        )

    def participant_tracker(self):
        return Agent(
            role="Meeting Participant Tracker",
            backstory=(
                "You have participated in many meetings and from "
                "the meeting transcription you can identify who "
                "actually attended the meeting. You know that a "
                "participant is someone who had spoken in the meeting, "
                "and not merely mentioned by other participants."
            ),
            goal="Create a list of names who participated in the meeting.",
            tools=[transcription_reader],
            allow_delegation=False,
            verbose=True,
            # llm=self.OpenAIGPT4,
            llm=self.llm_model,
        )

    def action_items_tracker(self):
        return Agent(
            role="Meeting Action Items Tracker",
            backstory=(
                "You have participated in many meetings and from "
                "the meeting transcription you can identify the "
                "future action items discussed by the participants. You "
                "understand that every meeting may not have any future "
                "action items, in which case you don't do anything. However, "
                "if any participant mentions about his or her plans for "
                "the future, include the plan against the name of the participant."
            ),
            goal="Create a list of action items discussed in the meeting.",
            tools=[transcription_reader],
            allow_delegation=False,
            verbose=True,
            # llm=self.OpenAIGPT4,
            llm=self.llm_model,
        )
