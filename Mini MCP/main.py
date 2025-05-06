# Mini MCP Demo: RAG + Memory + Tool Use

# STEP 1: Imports and Setup
from typing import List
import datetime

# Mock modules for vector search and tool use
class VectorDatabase:
    def search(self, query: str) -> List[str]:
        # Simulate search
        return [f"Doc: Return policy says 30 days from purchase"]

class MemoryStore:
    def __init__(self):
        self.past_interactions = {}
    
    def get_memory(self, user_id):
        return self.past_interactions.get(user_id, "")
    
    def update_memory(self, user_id, text):
        self.past_interactions[user_id] = text

class WeatherAPI:
    def get_forecast(self, location: str):
        # Simulate API call
        return f"Forecast for {location}: Rain expected tomorrow."

# STEP 2: MCP Components
class ContextAgent:
    def __init__(self, memory: MemoryStore, retriever: VectorDatabase, tools: dict):
        self.memory = memory
        self.retriever = retriever
        self.tools = tools

    def collect_context(self, user_id: str, query: str, location: str):
        memory = self.memory.get_memory(user_id)
        docs = self.retriever.search(query)
        weather = self.tools['weather'].get_forecast(location)
        
        return f"MEMORY:\n{memory}\n\nDOCUMENTS:\n{docs[0]}\n\nTOOL OUTPUT:\n{weather}"

# STEP 3: Simulated LLM Response
class FakeLLM:
    def generate(self, prompt: str):
        return f"[LLM Response]\nBased on context:\n{prompt}\n\nYour return window ends April 20. Also, rain is expected tomorrow."

# STEP 4: Controller
class MCPController:
    def __init__(self, agent: ContextAgent, llm: FakeLLM, memory: MemoryStore):
        self.agent = agent
        self.llm = llm
        self.memory = memory

    def handle_request(self, user_id: str, query: str, location: str):
        context = self.agent.collect_context(user_id, query, location)
        response = self.llm.generate(context)
        self.memory.update_memory(user_id, query)
        return response

# STEP 5: Run
if __name__ == "__main__":
    memory = MemoryStore()
    retriever = VectorDatabase()
    weather_tool = WeatherAPI()
    tools = {'weather': weather_tool}

    agent = ContextAgent(memory, retriever, tools)
    llm = FakeLLM()
    controller = MCPController(agent, llm, memory)

    # Simulate user interaction
    user_id = "user_123"
    query = "What’s my return window, and will it rain tomorrow?"
    location = "Bentonville"

    print(controller.handle_request(user_id, query, location))
