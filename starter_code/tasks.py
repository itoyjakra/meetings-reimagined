from textwrap import dedent

from crewai import Task


class MeetingTasks:
    def summarize(self, agent, num_words: int):
        return Task(
            description=f"Summarize the transcription of the meeting",
            expected_output=f"A {num_words} summary of the meeting.",
            agent=agent,
            tools=[],
            context=[],
            async_execution=True,
        )

    def track_participants(self, agent):
        return Task(
            description="Identify the participants of the meeting.",
            expected_output="A list of names.",
            agent=agent,
            tools=[],
            context=[],
            async_execution=True,
        )

    def collect_action_items(self, agent, context=[]):
        return Task(
            description="Identify the action items to be performed before the next meeting.",
            expected_output="A list of items.",
            agent=agent,
            tools=[],
            context=context,
            # async_execution=True,
        )

    def rag_search(self, agent, question):
        return Task(
            description="Answer the question.",
            expected_output=(
                f"An answer to the question: {question}, or a mention "
                "that documents don't contain the answer."
            ),
            agent=agent,
            tools=[],
            context=[],
            async_execution=False,
        )
