"""SQL Agent using LangGraph - follows agent-lightning APO pattern."""

from eval_utils import evaluate_query
from agentlightning.types import PromptTemplate
from agentlightning.litagent import rollout
from langgraph.graph import END, START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
import os
import re
from typing import Any, Dict, Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()


# Default prompts (WRITE_QUERY_PROMPT will be optimized by APO)
CHECK_QUERY_PROMPT = ChatPromptTemplate([
    ("system", """You are a SQL expert with a strong attention to detail.
Double check the {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins
- Explicit query execution failures
- Clearly unreasonable query execution results

## Table Schema ##

{table_info}

## Output Format ##

If any mistakes from the list above are found, list each error clearly.
After listing mistakes (if any), conclude with **ONE** of the following exact phrases in all caps and without surrounding quotes:
- If mistakes are found: `THE QUERY IS INCORRECT.`
- If no mistakes are found: `THE QUERY IS CORRECT.`

DO NOT write the corrected query in the response. You only need to report the mistakes.""".strip()),
    ("user", """Question: {input}

Query:

```{dialect}
{query}
```

Execution result:

```
{execution}
```"""),
])


REWRITE_QUERY_PROMPT = ChatPromptTemplate([
    ("system", """You are an agent designed to interact with a SQL database.
Rewrite the previous {dialect} query to fix errors based on the provided feedback.
The goal is to answer the original question.
Make sure to address all points in the feedback.

Pay attention to use only the column names that you can see in the schema description.
Be careful to not query for columns that do not exist.
Also, pay attention to which column is in which table.

## Table Schema ##

Only use the following tables:
{table_info}

## Output Format ##

Respond in the following format:

```{dialect}
REWRITTEN QUERY
```""".strip()),
    ("user", """Question: {input}

## Previous query ##

```{dialect}
{query}
```

## Previous execution result ##

```
{execution}
```

## Feedback ##

{feedback}

Please rewrite the query to address the feedback."""),
])


class State(MessagesState):
    question: str
    query: str
    execution: str
    answer: str
    feedback: str
    num_turns: int
    messages: list[AnyMessage]


class SQLAgent:
    """SQL agent with query generation, execution, checking, and rewriting."""

    def __init__(
        self,
        db_path: str,
        max_turns: int,
        write_prompt: str,
        model: str,
        temperature: float,
        table_info_truncate: int = 2048,
        execution_truncate: int = 2048,
        debug: bool = False,
    ):
        self.db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
        self.max_turns = max_turns
        self.table_info_truncate = table_info_truncate
        self.execution_truncate = execution_truncate
        self.debug = debug

        # Use the APO-provided write prompt
        self.write_query_prompt = ChatPromptTemplate([
            ("system", write_prompt),
            ("user", "Question: {input}")
        ])
        self.check_query_prompt = CHECK_QUERY_PROMPT
        self.rewrite_query_prompt = REWRITE_QUERY_PROMPT

        # Initialize LLM
        self.llm = init_chat_model(
            model,
            model_provider="openai",
            openai_api_base=os.environ.get(
                "OPENAI_API_BASE", "https://api.openai.com/v1"),
            openai_api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=temperature,
            max_tokens=2048,
        )
        self.query_tool = QuerySQLDatabaseTool(db=self.db)

    def get_table_info(self) -> str:
        """Get the table information in a human-readable format."""
        try:
            table_info = self.db.get_table_info()
            if len(table_info) > self.table_info_truncate:
                table_info = table_info[: self.table_info_truncate] + \
                    "\n... (truncated)"
            return table_info
        except Exception as e:
            print(f"Failed to get table info: {e}")
            return "No schema available."

    def invoke_prompt(self, prompt: Any) -> AnyMessage:
        """Invoke LLM with prompt and handle errors."""
        try:
            result = self.llm.invoke(prompt)
        except Exception as e:
            print(f"Failed to invoke prompt: {e}")

            result = self.llm.invoke(
                [HumanMessage(content="Please create a random SQL query as an example.")])
        return result

    def truncate_execution(self, execution: str) -> str:
        """Truncate the execution result to a reasonable length."""
        if len(execution) > self.execution_truncate:
            return execution[: self.execution_truncate] + "\n... (truncated)"
        return execution

    def parse_query(self, message: AnyMessage) -> str | None:
        """Parse SQL query from LLM response."""
        result: str | None = None

        for match in re.finditer(r".*```\w*\n(.*?)\n```.*", message.content, re.DOTALL):
            result = match.group(1).strip()
        return result

    def write_query(self, state: State) -> State:
        """Generate SQL query to fetch information."""
        prompt: Any = self.write_query_prompt.invoke({
            "dialect": self.db.dialect,
            "table_info": self.get_table_info(),
            "input": state["question"]
        })
        result = self.invoke_prompt(prompt)

        query = self.parse_query(result) or result.content
        return {
            **state,
            "query": query,
            "num_turns": 1,
            "messages": [*prompt.messages, result],
        }

    def execute_query(self, state: State) -> State:
        """Execute SQL query."""
        execution_result = self.query_tool.invoke(
            state["query"])
        if not isinstance(execution_result, str):

            execution_result = str(execution_result)
        return {**state, "execution": execution_result}

    def check_query(self, state: State) -> State:
        """Check the SQL query for correctness."""
        prompt: Any = self.check_query_prompt.invoke({
            "dialect": self.db.dialect,
            "table_info": self.get_table_info(),
            "input": state["question"],
            "query": state["query"],
            "execution": self.truncate_execution(state["execution"]),
        })
        result = self.invoke_prompt(prompt)

        res = {
            **state,
            "feedback": result.content,
            "messages": [*state.get("messages", []), *prompt.messages, result],
        }
        return res

    def rewrite_query(self, state: State) -> State:
        """Rewrite SQL query if necessary."""
        prompt: Any = self.rewrite_query_prompt.invoke({
            "dialect": self.db.dialect,
            "table_info": self.get_table_info(),
            "input": state["question"],
            "query": state["query"],
            "execution": self.truncate_execution(state["execution"]),
            "feedback": state["feedback"],
        })
        result = self.invoke_prompt(prompt)

        rewritten_query = self.parse_query(result)

        return {
            **state,
            "query": rewritten_query or state["query"],
            "num_turns": state.get("num_turns", 0) + 1,
            "messages": [*prompt.messages, result],
        }

    def should_continue(self, state: State) -> Literal[END, "rewrite_query"]:
        """Determine if the agent should continue based on the result."""
        if state["messages"] and isinstance(state["messages"][-1], BaseMessage):
            last_message = state["messages"][-1]
            if "THE QUERY IS CORRECT" in last_message.content:
                if "THE QUERY IS INCORRECT" in last_message.content:
                    correct_index = last_message.content.rfind(
                        "THE QUERY IS CORRECT")
                    incorrect_index = last_message.content.rfind(
                        "THE QUERY IS INCORRECT")
                    if correct_index > incorrect_index:
                        return END
                else:
                    return END

        if state.get("num_turns", 0) >= self.max_turns:
            return END

        return "rewrite_query"

    def graph(self):
        """Build the LangGraph workflow."""
        builder = StateGraph(State)
        builder.add_node(self.write_query)
        builder.add_node(self.execute_query)
        builder.add_node(self.check_query)
        builder.add_node(self.rewrite_query)

        builder.add_edge(START, "write_query")
        builder.add_edge("write_query", "execute_query")
        builder.add_edge("execute_query", "check_query")
        builder.add_conditional_edges(
            "check_query",
            self.should_continue,
        )
        builder.add_edge("rewrite_query", "execute_query")

        return builder.compile()


def prompt_template_baseline() -> PromptTemplate:
    """Baseline prompt template for APO to optimize."""
    return PromptTemplate(
        template="""You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run to help find the answer.

Pay attention to use only the column names that you can see in the schema description.
Be careful to not query for columns that do not exist.
Also, pay attention to which column is in which table.

## Table Schema ##

Only use the following tables:
{table_info}

## Output Format ##

Respond in the following format:

```{dialect}
GENERATED QUERY
```""",
        engine="f-string",
    )


@rollout
def sql_agent_rollout(task: Dict[str, Any], prompt_template: PromptTemplate) -> float:
    """SQL agent rollout following APO pattern from room_selector.py."""

    question = task["question"]
    db_id = task["db_id"]
    ground_truth = task["query"]

    db_path = os.path.join("databases", db_id, f"{db_id}.sqlite")

    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        return 0.0

    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    table_info = db.get_table_info()

    formatted_prompt = prompt_template.format(
        dialect="SQLite",
        table_info=table_info,
        input=question,
    )

    agent = SQLAgent(
        db_path=db_path,
        max_turns=3,
        write_prompt=formatted_prompt,
        model="gpt-5-mini",
        temperature=0.0,
    ).graph()

    try:
        result = agent.invoke(
            {"question": question},
            {"recursion_limit": 100}
        )
        predicted_query = result.get("query", "")
    except Exception as e:
        print(f"Agent execution failed: {e}")
        return 0.0

    # Evaluate
    reward = evaluate_query(predicted_query, ground_truth,
                            db_path, raise_on_error=False)

    print(f"Question: {question[:80]}...")
    print(f"Predicted: {predicted_query[:80]}...")
    print(f"Reward: {reward}")

    return reward
