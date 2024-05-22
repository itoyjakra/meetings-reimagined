"""Workflow to handle the set of tasks immediately following the completion of a meeting."""

import json
import os

import agentops
from crewai import Agent, Crew, Task
from crewai_tools import DirectoryReadTool, FileReadTool
from dotenv import load_dotenv
from llms import LLMConfig


class AfterMeetingCrew:
    """Crew to take care of tasks right after the meeting is over."""

    def __init__(
        self,
        agent_config_path: str,
        task_config_path: str,
        transcription_path: str,
        artifacts_path: str,
        llm_model,
    ) -> None:
        self.agent_config = json.loads(open(agent_config_path, "r").read())
        self.task_config = json.loads(open(task_config_path, "r").read())
        self.llm_model = llm_model
        self.transcription_reader = FileReadTool(file_path=transcription_path)
        self.artifacts_paths = artifacts_path

    @property
    def summarizer(self) -> Agent:
        """Summarizes the meeting transcription."""
        summarizer_config = self.agent_config["summarizer"]
        return Agent(
            **summarizer_config,
            verbose=True,
            llm=self.llm_model,
            tools=[self.transcription_reader],
        )

    @property
    def participant_tracker(self) -> Agent:
        """Tracks the meeting participants."""
        part_track_config = self.agent_config["participant_tracker"]
        return Agent(
            **part_track_config,
            verbose=True,
            llm=self.llm_model,
            tools=[self.transcription_reader],
        )

    @property
    def action_items_tracker(self) -> Agent:
        """Tracks action items from the meeting."""
        act_items_config = self.agent_config["action_items_tracker"]
        return Agent(
            **act_items_config,
            verbose=True,
            llm=self.llm_model,
            tools=[self.transcription_reader],
        )

    @property
    def qna_tracker(self) -> Agent:
        """Tracks all questions and answer pairs from the meeting."""
        qna_config = self.agent_config["qna_tracker"]
        return Agent(
            **qna_config,
            verbose=True,
            llm=self.llm_model,
            tools=[self.transcription_reader],
        )

    @property
    def track_qna(self) -> Task:
        """Q&A tracking task."""
        qna_config = self.task_config["track_qna"]
        return Task(
            **qna_config,
            verbose=True,
            agent=self.qna_tracker,
            output_file=f"{self.artifacts_paths}/qna.json",
        )

    @property
    def summarize(self) -> Task:
        """Summarization task."""
        summarize_config = self.task_config["summarize"]
        return Task(
            **summarize_config,
            verbose=True,
            agent=self.summarizer,
            output_file=f"{self.artifacts_paths}/summary.txt",
        )

    @property
    def track_participants(self) -> Task:
        """Participant tracking task."""
        track_part_config = self.task_config["track_participation"]
        return Task(
            **track_part_config,
            verbose=True,
            agent=self.participant_tracker,
            output_file=f"{self.artifacts_paths}/participants.txt",
        )

    @property
    def collect_action_items(self) -> Task:
        """Action items tracking task."""
        action_item_config = self.task_config["collect_action_items"]
        return Task(
            **action_item_config,
            verbose=True,
            agent=self.action_items_tracker,
            output_file=f"{self.artifacts_paths}/action_items.json",
        )

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
                self.summarize,
                self.track_participants,
                self.collect_action_items,
                self.track_qna,
            ],
            verbose=True,
        )

    def run(self) -> None:
        """Runs the crew execution."""
        # print(json.dumps(self.agent_config, indent=4))
        crew = self.build_crew()
        result = crew.kickoff()
        print(result)


if __name__ == "__main__":
    load_dotenv()
    agentops.init(
        tags=["crewai", "meeting", "after_meeting_tasks"],
        api_key=os.getenv("AGENTOPS_API_KEY"),
    )
    llm_config = LLMConfig(platform="openai", model="gpt-4o", temperature=0.5)
    llm_model = llm_config.get_llm_model()

    meeting_topic = "topic1"
    meeting_date = "2024-05-04"
    transcription_path = (
        f"meetings/{meeting_topic}/{meeting_date}/transcription/trans.txt"
    )
    artifacts_path = f"meetings/{meeting_topic}/{meeting_date}/artifacts"

    crew = AfterMeetingCrew(
        agent_config_path="config/agents.json",
        task_config_path="config/tasks.json",
        transcription_path=transcription_path,
        artifacts_path=artifacts_path,
        llm_model=llm_model,
    )
    result = crew.run()

    agentops.end_session("Success!")
