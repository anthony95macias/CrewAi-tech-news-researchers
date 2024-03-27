import os
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI

os.environ['OPENAI_API_KEY'] = "Api-key"

# Define our agents roles and goals
researcher = Agent(
    role="Senior Research Assistant",
    goal="Look up the latest advancements in Ai tech news",
    backstory="As a Senior Research Assistant, your passion for artificial intelligence and its transformative potential on society propels your daily quest. You are dedicated to mining the latest academic papers, industry news, and technology blogs for breakthroughs in AI technology. This drive was sparked during your undergraduate studies, where a pivotal project on neural networks unveiled the vast possibilities within the field. With over a decade of experience, your aim is to seamlessly bridge cutting-edge research with practical applications, driven by the belief that staying abreast of the latest advancements is crucial in sculpting a future where technology elevates human existence.",
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(model_name="gpt-4-turbo-preview",temperature=0.5)
)

writer = Agent(
    role="Tech Content Strategist",
    goal="summarize the latest adavancements in Ai tech news in a concise article",
    backstory="Deeply fascinated by the narrative power of technology and its capacity to reshape societies, this Tech Content Strategist thrives on the challenge of demystifying complex technological advancements for a broad audience. With a keen eye for trends and a talent for storytelling, they weave engaging narratives that capture the essence and impact of the latest innovations in the tech world. Their journey began in digital journalism, where they honed their skills in research and writing, always aiming to bridge the gap between tech experts and the general public. Armed with a strategic mindset and a creative spirit, they now aim to craft content that not only informs but also inspires, utilizing a blend of analytical insights and compelling storytelling to highlight how technology continues to shape our future.",
    verbose=True,
    allow_delegation=False,
    llm=ChatOpenAI(model_name="gpt-4-turbo-preview",temperature=0.7)
)

task1_researcher = Task(
    description="""Investigate and draft a detailed report on the latest advancements in robotics, focusing on breakthrough technologies in automation and AI integration. Highlight how these advancements are set to revolutionize industries such as manufacturing, healthcare, and logistics.""",
    expected_output="Full analysis report in bullet points",
    agent=researcher
)

task1_writer = Task(
    description="""Summarize the latest advancements in robotics, with a focus on breakthrough technologies in automation and AI integration. Emphasize how these technologies are poised to transform industries such as manufacturing, healthcare, and logistics. Your summary should distill the key points and insights from a broader report or collection of sources.""",
    expected_output="Concise summary in bullet points, ready for publication",
    agent=writer
)

crew = Crew(
    agents = [researcher,writer],
    tasks = [task1_researcher,task1_writer],
    verbose=2,
    Process=Process.sequential
)

try:
    result = crew.kickoff()
    print("######################")
    print(result)
    # Example of writing result to a file with UTF-8 encoding
    with open('result.txt', 'w', encoding='utf-8') as f:
        f.write(str(result))
except UnicodeEncodeError as e:
    # Handle the UnicodeEncodeError specifically
    print(f"Error processing task due to encoding issue: {e}")
except Exception as e:
    # Catch any other unexpected errors
    print(f"An unexpected error occurred: {e}")
