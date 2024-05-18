from crewai import Task
from textwrap import dedent


class MeetingTasks:
    def summarize(self, agent, num_words: int):
        return Task(
            description=f"Summarize the transcription of the meeting",
            expected_output=f"A {num_words} summary of the meeting.",
            agent=agent,
            tools=[],
            context=[],
        )

    def track_participants(self, agent):
        return Task(
            description="Identify the participants of the meeting.",
            expected_output="A list of names.",
            agent=agent,
            tools=[],
            context=[]
        )

    def collect_action_items(self, agent):
        return Task(
            description="Identify the action items to be performed before the next meeting.",
            expected_output="A list of items.",
            agent=agent,
            tools=[],
            context=[]
        )