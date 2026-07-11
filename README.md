# Student Enrollment Assistant

A conversational AI agent for a university admissions office. It helps prospective students with program information, application deadlines, and application status by calling tools — not just relying on LLM knowledge.

## How It Works

The agent uses **LangChain** with **OpenRouter** (GPT-4o-mini) and three tools:

| Tool | Input | Returns |
|------|-------|---------|
| `get_program_info` | program name | Duration, tuition, prerequisites, pending documents |
| `check_application_status` | applicant ID | Status, next step |
| `get_deadlines` | program name | Application, document, and decision deadlines |

The agent loop receives a user message, reasons about which tool(s) to call, executes them, and formulates a natural language response. It supports multi-turn conversation and remembers context within a session.

**Graceful escalation:** If the agent can't answer with its tools (e.g., fee waivers), it suggests speaking with an enrollment counselor.

## Project Structure

```
enrollment_agent.py   # Agent, tools, prompt, and interactive chat
mock_data.py          # Hardcoded mock data (programs, deadlines, applicants)
.env                  # API keys (not committed)
.env.example          # Template for .env
requirements.txt      # Python dependencies
```

## Setup

**1. Clone and create virtual environment:**

```bash
git clone <repo-url>
cd edmo-task
python -m venv .venv
```

**2. Activate virtual environment:**

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

**3. Install dependencies:**

```bash
pip install langchain langchain-core langchain-openai langchain-community langgraph langfuse python-dotenv
```

**4. Create `.env` file:**

```bash
cp .env.example .env
```

Then edit `.env` and add your OpenRouter API key:

```
OPENAI_API_KEY=your-openrouter-api-key-or-openai-key
```

To enable Langfuse tracing (optional), uncomment and fill in the Langfuse keys in `.env`.

## Run

**Interactive chat:**

```bash
py enrollment_agent.py
```

Type your questions and `quit` to exit.
