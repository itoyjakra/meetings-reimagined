"""Workflow to handle the set of tasks immediately following the completion of a meeting."""

import json
import os
from textwrap import dedent

import agentops
from agents import MeetingAgents
from crewai import Agent, Crew, Process, Task
from dotenv import load_dotenv

# from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from llms import LLMConfig
from tasks import MeetingTasks


class AfterMeetingCrew:
    """Crew to take care of tasks right after the meeting is over."""

    def __init__(
        self,
        agent_config_path: str,
        task_config_path: str,
        transcription_path: str,
        llm_model,
    ) -> None:
        self.agent_config = json.loads(open(agent_config_path, "r").read())
        self.task_config = json.loads(open(task_config_path, "r").read())
        self.transcription_path = transcription_path
        self.llm_model = llm_model

    @property
    def summarizer(self) -> Agent:
        """Summarizes the meeting transcription."""
        summarizer_config = self.agent_config["summarizer"]
        return Agent(**summarizer_config, verbose=True, llm=self.llm_model)

    @property
    def participant_tracker(self) -> Agent:
        """Tracks the meeting participants."""
        part_track_config = self.agent_config["participant_tracker"]
        return Agent(**part_track_config, verbose=True, llm=self.llm_model)

    @property
    def action_items_tracker(self) -> Agent:
        """Tracks action items from the meeting."""
        act_items_config = self.agent_config["action_items_tracker"]
        return Agent(**act_items_config, verbose=True, llm=self.llm_model)

    @property
    def qna_tracker(self) -> Agent:
        """Tracks all questions and answer pairs from the meeting."""
        qna_config = self.agent_config["qna_tracker"]
        return Agent(**qna_config, verbose=True, llm=self.llm_model)

    @property
    def track_qna(self) -> Task:
        """"""
        qna_config = self.task_config["qna_tracker"]
        return Task(**qna_config, verbose=True, llm=self.llm_model)

    def build_crew(self) -> Crew:
        """Builds the crew for the job."""
        return Crew(
            agents=[
                self.summarizer,
                self.participant_tracker,
                self.action_items_tracker,
                self.qna_tracker,
            ],
            tasks=[
                summarize_task,
                track_participants_task,
                collect_action_items,
                get_answers,
            ],
            verbose=True,
        )

    def run(self):
        """Runs the crew execution."""
        print(json.dumps(self.agent_config, indent=4))
        pass


if __name__ == "__main__":
    load_dotenv()
    agentops.init(tags=["crewai", "meeting"], api_key=os.getenv("AGENTOPS_API_KEY"))
    llm_config = LLMConfig(platform="openai", model="gpt-4o")
    llm_model = llm_config.get_llm_model()

    crew = AfterMeetingCrew(
        agent_config_path="config/agents.json",
        task_config_path="config/tasks.json",
        transcription_path="meetings/topic1/2024-05-04.txt",
        llm_model=llm_model,
    )
    result = crew.run()
