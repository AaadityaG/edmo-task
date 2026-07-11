import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langfuse import Langfuse
from langfuse.langchain import CallbackHandler
from mock_data import PROGRAMS, DEADLINES, APPLICANTS

load_dotenv()

langfuse = Langfuse()
langfuse_handler = CallbackHandler()

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7,
)


# --- Tools ---
@tool
def get_program_info(program_name: str) -> str:
    """Get detailed information about an academic program including duration, tuition, and prerequisites.
    Example: "computer science" or "data science"
    """
    with langfuse.start_as_current_observation(as_type="tool", name="get_program_info", input=program_name) as obs:
        key = program_name.lower().strip()
        if key in PROGRAMS:
            result = PROGRAMS[key]
        else:
            result = f"Program '{program_name}' not found. Available programs: {', '.join(PROGRAMS.keys())}"
        obs.update(output=result)
        return str(result)


@tool
def check_application_status(applicant_id: str) -> str:
    """Check the status of a student's application using their applicant ID.
    Example: "APP-1042"
    """
    with langfuse.start_as_current_observation(as_type="tool", name="check_application_status", input=applicant_id) as obs:
        id_upper = applicant_id.upper().strip()
        if id_upper in APPLICANTS:
            result = APPLICANTS[id_upper]
        else:
            result = f"Applicant '{applicant_id}' not found. Please check your ID and try again."
        obs.update(output=result)
        return str(result)


@tool
def get_deadlines(program_name: str) -> str:
    """Get application deadlines for a specific program including application, document submission, and decision dates.
    Example: "computer science" or "data science"
    """
    with langfuse.start_as_current_observation(as_type="tool", name="get_deadlines", input=program_name) as obs:
        key = program_name.lower().strip()
        if key in DEADLINES:
            result = DEADLINES[key]
        else:
            result = f"Deadline information for '{program_name}' not found."
        obs.update(output=result)
        return str(result)


tools = [get_program_info, check_application_status, get_deadlines]
tool_map = {t.name: t for t in tools}

llm_with_tools = llm.bind_tools(tools)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a University Enrollment Assistant. You MUST use tools to answer questions. Never answer from your own knowledge.

Tools available:
- get_program_info(program_name): Call when student asks about a program, its details, OR required documents. Valid: "computer science", "data science", "business administration"
- check_application_status(applicant_id): Call when student asks about application status or gives an ID like "APP-1042"
- get_deadlines(program_name): Call when student asks about deadlines for a program

Behavior:
- On greetings (hi, hello): Introduce yourself and list what you can help with
- Vague questions (e.g. "what programs do you offer"): Ask which specific program
- Questions about a specific program, status, deadlines, or documents: ALWAYS call the matching tool
- Questions you cannot answer with tools: "I'd recommend speaking with an enrollment counselor for that. Would you like me to connect you?"
- Remember the student's applicant ID if they give it"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chat_history: list = []


def run_agent(user_input: str) -> str:
    with langfuse.start_as_current_observation(name="Enrollment Agent Run") as agent_span:
        messages = prompt.invoke({"input": user_input, "history": chat_history})
        messages = messages.to_messages()

        response = llm_with_tools.invoke(messages, config={"callbacks": [langfuse_handler]})
        # print(f"  [DEBUG] tool_calls: {response.tool_calls}")

        while response.tool_calls:
            # print(f"  [Calling tools: {[tc['name'] for tc in response.tool_calls]}]")
            messages.append(response)
            for tc in response.tool_calls:
                tool_fn = tool_map[tc["name"]]
                result = tool_fn.invoke(tc["args"])
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
            response = llm_with_tools.invoke(messages, config={"callbacks": [langfuse_handler]})
            # print(f"  [DEBUG] tool_calls: {response.tool_calls}")

        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response.content))
        return response.content


# --- Interactive Chat ---
if __name__ == "__main__":
    print("Enrollment Assistant (type 'quit' to exit)")
    print()

    while True:
        user_input = input("You: ")
        if user_input.lower() in ("quit", "exit", "q"):
            langfuse.flush()
            print("Goodbye!")
            break
        response = run_agent(user_input)
        print(f"Agent: {response}\n")



