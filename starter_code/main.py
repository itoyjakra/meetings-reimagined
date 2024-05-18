import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
# from decouple import config

from textwrap import dedent
from agents import MeetingAgents
from tasks import MeetingTasks
from dotenv import load_dotenv

# Install duckduckgo-search for this example:
# !pip install -U duckduckgo-search

# from langchain.tools import DuckDuckGoSearchRun

# search_tool = DuckDuckGoSearchRun()

# os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")
# os.environ["OPENAI_ORGANIZATION"] = config("OPENAI_ORGANIZATION_ID")

# This is the main class that you will use to define your custom crew.
# You can define as many agents and tasks as you want in agents.py and tasks.py


class MeetingCrew:
    def __init__(self, summary_num_words:int):
        self.summary_num_words = summary_num_words

    def run(self):
        # Define your custom agents and tasks in agents.py and tasks.py
        agents = MeetingAgents()
        tasks = MeetingTasks()

        # Define your custom agents and tasks here
        summarizer_agent = agents.summarizer()
        participant_tracker_agent = agents.participant_tracker()
        action_item_tracker_agent = agents.action_items_tracker()

        # Custom tasks include agent name and variables as input
        summarize_task = tasks.summarize(
            summarizer_agent,
            self.summary_num_words,
        )

        track_participants_task = tasks.track_participants(
            participant_tracker_agent,
        )

        collect_action_items = tasks.collect_action_items(
            action_item_tracker_agent
        )

        # Define your custom crew here
        crew = Crew(
            agents=[summarizer_agent, participant_tracker_agent, action_item_tracker_agent],
            tasks=[summarize_task, track_participants_task, collect_action_items],
            verbose=True,
        )

        result = crew.kickoff()
        return result


# This is the main function that you will use to run your custom crew.
if __name__ == "__main__":
    load_dotenv()

    print("## Welcome to Crew AI Template")
    print("-------------------------------")
    num_words = input(dedent("""Enter number of words in meeting summary: """))

    crew = MeetingCrew(num_words)
    result = crew.run()
    print("\n\n########################")
    print("## Here is your meeting crew run result:")
    print("########################\n")
    print(result)