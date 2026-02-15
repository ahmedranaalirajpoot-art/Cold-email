import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable
from crewai import Agent, Task, Crew, LLM
from crewai_tools import ScrapeWebsiteTool
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Service:
    """Represents an agency service offering."""
    name: str
    description: str
    best_for: str


@dataclass
class AgentConfig:
    """Configuration for creating an agent dynamically."""
    role: str
    goal_template: str
    backstory: str
    tools: List = field(default_factory=list)


@dataclass
class TaskConfig:
    """Configuration for creating a task dynamically."""
    description_template: str
    expected_output_template: str


class AgencyConfig:
    """Dynamic configuration for the agency."""
    
    def __init__(self):
        self.services: List[Service] = []
        self.llm_model: str = "gemini/gemini-2.5-flash"
        self.api_key: Optional[str] = os.getenv("GEMINI_API_KEY")
        self._setup_default_services()
    
    def _setup_default_services(self):
        """Initialize with default services - can be modified at runtime."""
        self.services = [
            Service(
                name="Increase Organic Reach",
                description="We increase your website visibility through SEO optimization.",
                best_for="websites looking to improve search rankings"
            ),
            Service(
                name="Website Fixing",
                description="We repair and update old websites to modern standards.",
                best_for="old or broken websites needing technical repairs"
            ),
            Service(
                name="Content Writing",
                description="We create engaging, SEO-optimized content for your website.",
                best_for="new websites or those needing fresh content"
            )
        ]
    
    def add_service(self, name: str, description: str, best_for: str):
        """Add a new service at runtime."""
        self.services.append(Service(name, description, best_for))
    
    def remove_service(self, name: str):
        """Remove a service by name."""
        self.services = [s for s in self.services if s.name != name]
    
    def get_services_text(self) -> str:
        """Generate formatted services description."""
        return "\n".join([
            f"{i+1}. {s.name}: {s.description} (Best for: {s.best_for})"
            for i, s in enumerate(self.services)
        ])


class AgentFactory:
    """Factory for creating agents dynamically based on configuration."""
    
    def __init__(self, llm: LLM, scraping_tool: ScrapeWebsiteTool):
        self.llm = llm
        self.scraping_tool = scraping_tool
    
    def create_agent(self, config: AgentConfig, context: Dict[str, str]) -> Agent:
        """Create an agent with dynamic goal based on context."""
        goal = config.goal_template.format(**context)
        
        return Agent(
            role=config.role,
            goal=goal,
            backstory=config.backstory,
            tools=config.tools if config.tools else [self.scraping_tool],
            llm=self.llm
        )


class TaskFactory:
    """Factory for creating tasks dynamically based on configuration."""
    
    def create_task(self, config: TaskConfig, agent: Agent, context: Dict[str, str]) -> Task:
        """Create a task with dynamic description based on context."""
        description = config.description_template.format(**context)
        expected_output = config.expected_output_template.format(**context)
        
        return Task(
            description=description,
            expected_output=expected_output,
            agent=agent
        )


class ColdEmailWorkflow:
    """Main workflow orchestrator for the cold email generation system."""
    
    AGENT_CONFIGS = {
        "scraper": AgentConfig(
            role="Web Scraper",
            goal_template="Scrape relevant data from {website_url} to understand the business",
            backstory="You are an expert web scraper that extracts comprehensive data from websites. "
                     "You identify business type, services offered, target audience, and potential pain points."
        ),
        "strategist": AgentConfig(
            role="Service Strategist",
            goal_template="Analyze scraped data from {website_url} and recommend the best service from: {services}",
            backstory="You are a strategic consultant who analyzes websites and matches them with the perfect service. "
                     "You understand business needs and can articulate why a specific service would benefit them."
        ),
        "email_writer": AgentConfig(
            role="Professional Email Writer",
            goal_template="Write a compelling cold email to the owner of {website_url} proposing our services",
            backstory="You are an expert copywriter specializing in cold outreach. "
                     "You write personalized, persuasive emails that get responses. "
                     "You understand the prospect's business and tailor the message accordingly."
        )
    }
    
    TASK_CONFIGS = {
        "scraper": TaskConfig(
            description_template="""
            Scrape and analyze {website_url} thoroughly. Extract:
            - Business name and description
            - Products/services offered
            - Target audience
            - Current website quality indicators
            - Any visible pain points or improvement opportunities
            
            Provide a comprehensive summary of findings.
            """,
            expected_output_template="Detailed analysis of {website_url} including business overview and potential improvement areas"
        ),
        "strategist": TaskConfig(
            description_template="""
            Based on the scraped data from {website_url}, analyze which of our services would be most beneficial:
            
            Available Services:
            {services}
            
            Recommend the best service and explain WHY it would help this specific business.
            Include specific observations from the website analysis.
            """,
            expected_output_template="Recommended service with detailed justification tailored to {website_url}"
        ),
        "email_writer": TaskConfig(
            description_template="""
            Write a professional cold email to the website owner of {website_url}.
            
            Context:
            - Website analyzed: {website_url}
            - Recommended service: Use the strategist's recommendation
            - Our services: {services}
            
            The email should:
            - Be personalized based on the website analysis
            - Show genuine interest in their business
            - Present the recommended service as a solution to their specific needs
            - Include a clear call-to-action
            - Be professional yet conversational
            """,
            expected_output_template="Professional cold email ready to send to {website_url} owner"
        )
    }
    
    def __init__(self, config: Optional[AgencyConfig] = None):
        self.config = config or AgencyConfig()
        self.scraping_tool = ScrapeWebsiteTool()
        self.llm = LLM(
            model=self.config.llm_model,
            api_key=self.config.api_key
        )
        self.agent_factory = AgentFactory(self.llm, self.scraping_tool)
        self.task_factory = TaskFactory()
    
    def get_user_input(self) -> Dict[str, str]:
        """Collect all user inputs interactively."""
        print("=" * 60)
        print("COLD EMAIL GENERATOR - REAL-TIME CONFIGURATION")
        print("=" * 60)
        
        # Get website URL
        website_url = input("\nEnter the website URL to analyze: ").strip()
        while not website_url:
            print("Website URL is required!")
            website_url = input("Enter the website URL to analyze: ").strip()
        
        # Show current services
        print("\n" + "-" * 60)
        print("CURRENT SERVICES:")
        print("-" * 60)
        print(self.config.get_services_text())
        
        # Ask if user wants to modify services
        modify = input("\nWould you like to add/remove services? (yes/no): ").lower().strip()
        
        while modify in ['yes', 'y']:
            action = input("Add or remove service? (add/remove/done): ").lower().strip()
            
            if action == 'add':
                name = input("Service name: ").strip()
                description = input("Service description: ").strip()
                best_for = input("Best for (target audience): ").strip()
                if name and description:
                    self.config.add_service(name, description, best_for)
                    print(f"Added: {name}")
            
            elif action == 'remove':
                name = input("Service name to remove: ").strip()
                self.config.remove_service(name)
                print(f"Removed: {name}")
            
            elif action == 'done':
                break
            
            print("\nUpdated services:")
            print(self.config.get_services_text())
        
        # Get optional customizations
        print("\n" + "-" * 60)
        print("OPTIONAL CUSTOMIZATIONS:")
        print("-" * 60)
        
        tone = input("Email tone (professional/friendly/urgent) [default: professional]: ").strip()
        tone = tone if tone else "professional"
        
        custom_note = input("Any specific points to mention in the email (optional): ").strip()
        
        return {
            "website_url": website_url,
            "services": self.config.get_services_text(),
            "tone": tone,
            "custom_note": custom_note
        }
    
    def create_agents(self, context: Dict[str, str]) -> Dict[str, Agent]:
        """Create all agents dynamically."""
        return {
            name: self.agent_factory.create_agent(config, context)
            for name, config in self.AGENT_CONFIGS.items()
        }
    
    def create_tasks(self, agents: Dict[str, Agent], context: Dict[str, str]) -> List[Task]:
        """Create all tasks dynamically."""
        tasks = []
        task_order = ["scraper", "strategist", "email_writer"]
        
        for task_name in task_order:
            agent = agents[task_name]
            config = self.TASK_CONFIGS[task_name]
            task = self.task_factory.create_task(config, agent, context)
            tasks.append(task)
        
        return tasks
    
    def run(self) -> str:
        """Execute the complete workflow."""
        try:
            # Get user input
            context = self.get_user_input()
            
            # Create agents and tasks dynamically
            print("\n" + "=" * 60)
            print("CREATING AGENTS AND TASKS...")
            print("=" * 60)
            
            agents = self.create_agents(context)
            tasks = self.create_tasks(agents, context)
            
            # Create and run crew
            crew = Crew(
                agents=list(agents.values()),
                tasks=tasks,
                memory=True,
                verbose=True
            )
            
            print("\n" + "=" * 60)
            print("RUNNING WORKFLOW...")
            print("=" * 60 + "\n")
            
            result = crew.kickoff()
            return result
            
        except Exception as e:
            return f"Error during workflow execution: {str(e)}"


def main():
    """Main entry point."""
    workflow = ColdEmailWorkflow()
    result = workflow.run()
    
    print("\n" + "=" * 60)
    print("RESULT:")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
