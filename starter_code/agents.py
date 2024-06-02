import os
import re

from crewai import Agent
from crewai_tools import DirectoryReadTool, FileReadTool
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from loguru import logger

transcription_reader = FileReadTool(
    file_path="./meetings/topic1/2024-05-11/transcription/trans.txt"
)
directory_reader = DirectoryReadTool(directory="./meetings/topic1")


class MeetingAgents:
    def __init__(self, llm_model):
        self.llm_model = llm_model

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

    def answer_finder(self):
        return Agent(
            role="Finds answers from documents.",
            backstory=(
                "You are an expert RAG agent who finds the answer to the "
                "question. If the question is ambiguous you say so. If the "
                "answer is not found in any of the documents provided to you "
                "you say so. When you provide an answer, it is always grounded "
                "in one of the documents and you can point the user to the "
                "relevant section of the document."
            ),
            goal="Provides an answer grounded in the documents to the question.",
            tools=[directory_reader, transcription_reader],
            allow_delegation=False,
            verbose=True,
            llm=self.llm_model,
        )
