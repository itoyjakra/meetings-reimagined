import os
from textwrap import dedent

from agents import MeetingAgents
from crewai import Agent, Crew, Process, Task
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tasks import MeetingTasks

# Install duckduckgo-search for this example:
# !pip install -U duckduckgo-search

# from langchain.tools import DuckDuckGoSearchRun

# search_tool = DuckDuckGoSearchRun()


class MeetingCrew:
    def __init__(self, summary_num_words: int):
        self.summary_num_words = summary_num_words

    def run(self):
        agents = MeetingAgents()
        tasks = MeetingTasks()

        # Agents
        summarizer_agent = agents.summarizer()
        participant_tracker_agent = agents.participant_tracker()
        action_item_tracker_agent = agents.action_items_tracker()

        # Tasks
        summarize_task = tasks.summarize(
            summarizer_agent,
            self.summary_num_words,
        )

        track_participants_task = tasks.track_participants(
            participant_tracker_agent,
        )

        collect_action_items = tasks.collect_action_items(action_item_tracker_agent)

        # Crew
        crew = Crew(
            agents=[
                summarizer_agent,
                participant_tracker_agent,
                action_item_tracker_agent,
            ],
            tasks=[summarize_task, track_participants_task, collect_action_items],
            verbose=True,
        )

        result = crew.kickoff()
        return result


if __name__ == "__main__":
    load_dotenv()

    print("## Welcome to Crew AI Template")
    print("-------------------------------")
    num_words = input(dedent("""Enter number of words in meeting summary: """))
    num_words = int(num_words)

    crew = MeetingCrew(num_words)
    result = crew.run()
    print("\n\n########################")
    print("## Here is the meeting crew run result:")
    print("########################\n")
    print(result)
