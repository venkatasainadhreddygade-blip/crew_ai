from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
import os

# import crewai.llms.cache as _crewai_cache
# _crewai_cache.mark_cache_breakpoint = lambda msg: msg

load_dotenv()





# LLM Configuration
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)


@CrewBase
class MarketingCrewai:

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def research_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["research_agent"],
            tools=[SerperDevTool()],
            llm=llm,
            verbose=True
    )

    @agent
    def writer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["writer_agent"],
            llm=llm,
            verbose=True,
            allow_delegation=False,
            use_system_prompt=False
        )

    @agent
    def reviewer_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["reviewer_agent"],
            llm=llm,
            verbose=True,
            allow_delegation=False,
            use_system_prompt=False
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],
            agent=self.research_agent(),
            verbose=True
        )

    @task
    def writing_task(self) -> Task:
        return Task(
            config=self.tasks_config["writing_task"],
            agent=self.writer_agent(),
            output_file="report_output.md",
            verbose=True
        )

    @task
    def reviewing_task(self) -> Task:
        return Task(
            config=self.tasks_config["reviewing_task"],
            agent=self.reviewer_agent(),
            output_file="improvements_output.md",
            verbose=True
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            chat_llm=llm,
            verbose=True,
            memory=False
        )