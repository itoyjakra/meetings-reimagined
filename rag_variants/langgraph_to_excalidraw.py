from pydoc import doc

import langgraph
from dotenv import load_dotenv

load_dotenv()

from crewai import Crew
from crewai_agents import DrawingAgents
from crewai_tools import FileReadTool
from tasks import StudyTasks


def main():
    load_dotenv()

    print("## Welcome to the Crew that converts your LangGraph graph to Excalidraw!")
    print("-------------------------------")

    langgraph_file = input("Enter the langgraph file path\n")
    # file_read_tool = FileReadTool(file_path=langgraph_file, async_execution=False)

    # Available Agents
    agents = DrawingAgents()
    json_agent = agents.json_agent()
    excal_agent = agents.excalidraw_agent()

    # Create Tasks
    tasks = StudyTasks()
    study_lg = tasks.langgraph_document_study_task(json_agent)
    study_excal = tasks.excal_document_study_task(excal_agent)

    # Create contexts for the tasks
    study_lg.context = []
    study_excal.context = []

    # Create Crew
    crew = Crew(
        agents=[
            excal_agent,
            json_agent,
        ],
        tasks=[study_lg, study_excal],
    )

    play = crew.kickoff()

    # Print results
    print("\n\n################################################")
    print("## Here is the result")
    print("################################################\n")
    print(play)


if __name__ == "__main__":
    main()
