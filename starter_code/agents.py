from crewai import Agent
from textwrap import dedent
from langchain.llms import OpenAI, Ollama
from langchain_openai import ChatOpenAI
from crewai_tools import FileReadTool


transcription_reader = FileReadTool(file_path="./meetings/topic1/2024-05-04.txt")

class MeetingAgents:
    def __init__(self):
        self.OpenAIGPT35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        self.OpenAIGPT4 = ChatOpenAI(model_name="gpt-4", temperature=0.7)
        self.Ollama = Ollama(model="openhermes")

    def summarizer(self):
        return Agent(
            role="Document Summarizer",
            backstory=("You have participated in many technical meetings. "
                       "You can identify the topic of the discussion and "
                       "create a summary of the meeting capturing the "
                       "essential topics. "
            ),
            goal="Create a summary of the meeting without any individual's name.",
            tools=[transcription_reader],
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT4,
        )

    def participant_tracker(self):
        return Agent(
            role="Meeting Participant Tracker",
            backstory=("You have participated in many meetings and from "
                       "the meeting transcription you can identify who "
                       "actually attended the meeting. You know that a "
                       "participant is someone who had spoken in the meeting, "
                       "and not merely mentioned by other participants."
                      ),
            goal="Create a list of names who participated in the meeting.",
            tools=[transcription_reader],
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT4
        )

    def action_items_tracker(self):
        return Agent(
            role="Meeting Action Items Tracker",
            backstory=("You have participated in many meetings and from "
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
            llm=self.OpenAIGPT4
        )
