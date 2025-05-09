### 📘 README.md – *Mini Model Context Protocol (MCP) Demo*

#### 🚀 What is MCP?

**Model Context Protocol (MCP)** extends the concept of Retrieval-Augmented Generation (RAG) by orchestrating multiple external contexts:
- Past memory (long-term history)
- Retrieved documents (RAG)
- Tool outputs (APIs, DB queries)

More can be found here-https://modelcontextprotocol.io/introduction

It enables **agentic behavior** in LLMs, letting them pull and combine relevant information before generating responses.

### Diagram
<p align="center">
  <img src="./images/MCP.png" alt="MCP Diagram" width="600"/>
</p>


---

#### 🧱 What’s in This Repo?

This demo shows a simplified MCP-style system using:

| Component       | Description                              |
|----------------|------------------------------------------|
| `MemoryStore`   | Stores/retrieves user’s previous queries |
| `VectorDatabase`| Simulates document retrieval (RAG)       |
| `WeatherAPI`    | Simulates an external tool/API           |
| `ContextAgent`  | Collects all external context            |
| `FakeLLM`       | Mocks an LLM response using context      |
| `MCPController` | Coordinates the entire process           |

---

#### 🧪 Example Output

```
[LLM Response]
Based on context:
MEMORY:
What’s my return window, and will it rain tomorrow?

DOCUMENTS:
Doc: Return policy says 30 days from purchase

TOOL OUTPUT:
Forecast for Bentonville: Rain expected tomorrow.

Your return window ends April 20. Also, rain is expected tomorrow.
```

---

#### 🛠️ Run the Demo

```bash
python main.py
```

---

#### 💡 Future Additions

You could enhance this demo by:
- Using **LangChain** or **LlamaIndex** for real vector DB/RAG
- Integrating **OpenWeatherMap API** or other external tools
- Replacing `FakeLLM` with an actual LLM like GPT-4 via API
- Adding memory persistence (Redis, SQLite)

---



Feel free to fork and experiment with your own MCP extensions!
