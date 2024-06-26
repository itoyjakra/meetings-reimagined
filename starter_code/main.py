import os
from textwrap import dedent

import agentops
from agents import MeetingAgents
from crewai import Agent, Crew, Process, Task
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from llms import LLMConfig
from tasks import MeetingTasks

# Install duckduckgo-search for this example:
# !pip install -U duckduckgo-search

# from langchain.tools import DuckDuckGoSearchRun

# search_tool = DuckDuckGoSearchRun()


class MeetingCrew:
    def __init__(self, llm, summary_num_words: int = 200):
        self.llm = llm
        self.summary_num_words = summary_num_words

    def run(self, question):
        agents = MeetingAgents(llm_model=self.llm)
        tasks = MeetingTasks()

        # Agents
        summarizer_agent = agents.summarizer()
        participant_tracker_agent = agents.participant_tracker()
        action_item_tracker_agent = agents.action_items_tracker()
        rag_agent = agents.answer_finder()

        # Tasks
        summarize_task = tasks.summarize(
            summarizer_agent,
            self.summary_num_words,
        )

        track_participants_task = tasks.track_participants(
            participant_tracker_agent,
        )

        collect_action_items = tasks.collect_action_items(
            action_item_tracker_agent, context=[summarize_task, track_participants_task]
        )

        get_answers = tasks.rag_search(rag_agent, question=question)

        # Crew
        crew = Crew(
            agents=[
                summarizer_agent,
                participant_tracker_agent,
                action_item_tracker_agent,
                rag_agent,
            ],
            tasks=[
                summarize_task,
                track_participants_task,
                collect_action_items,
                get_answers,
            ],
            verbose=True,
        )

        result = crew.kickoff()
        return result


if __name__ == "__main__":
    load_dotenv()
    if os.getenv("AGENTOPS_API_KEY"):
        agentops.init(tags=["crewai", "meeting"], api_key=os.getenv("AGENTOPS_API_KEY"))

    #llm_config = LLMConfig(platform="openai", model="gpt-4o")
    llm_config = LLMConfig(
        platform="bedrock", model="anthropic.claude-3-sonnet-20240229-v1:0"
    )
    llm_model = llm_config.get_llm_model()

    print("## Welcome to Crew AI Template")
    print("-------------------------------")
    num_words = input(dedent("""Enter number of words in meeting summary: """))
    num_words = int(num_words)

    crew = MeetingCrew(llm=llm_model, summary_num_words=num_words)
    qq = "when is instruction fine tuning required?"
    qq = "What is PEFT?"
    # qq = "When was Andrew born?"
    result = crew.run(question=qq)
    print("\n\n########################")
    print("## Here is the meeting crew run result:")
    print("########################\n")
    print(result)
