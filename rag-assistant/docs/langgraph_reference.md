# LangGraph Reference Guide

LangGraph is a library built on top of LangChain for building stateful, multi-actor applications with LLMs. It models agentic workflows as directed graphs where nodes are functions and edges define the execution order.

## Core Abstractions

### StateGraph
`StateGraph` is the main class you use to define a workflow. You pass it a state schema (a Pydantic model or TypedDict), register nodes and edges, set an entry point, and then compile it.

```python
from langgraph.graph import StateGraph, END
from pydantic import BaseModel

class MyState(BaseModel):
    question: str
    answer: str = ""
    retries: int = 0

workflow = StateGraph(MyState)
```

### Nodes
A node is any callable that accepts the current state and returns a dictionary of fields to update.

```python
def my_node(state: MyState) -> dict:
    # process the state
    return {"answer": "computed answer"}

workflow.add_node("my_node", my_node)
```

### Edges
Edges connect nodes. A regular edge always goes from A to B. A conditional edge chooses the next node based on a routing function.

```python
# Regular edge
workflow.add_edge("node_a", "node_b")

# Conditional edge
def route(state: MyState) -> str:
    if state.retries < 3:
        return "retry"
    return "give_up"

workflow.add_conditional_edges(
    "grading_node",
    route,
    {
        "retry":    "retrieval_node",
        "give_up":  "fallback_node",
    }
)
```

### Entry Point and END
Every graph needs an entry point and at least one path to the special `END` sentinel.

```python
workflow.set_entry_point("query_analysis")
workflow.add_edge("generation", END)
```

### Compilation
Compiling the graph validates the topology and returns a `CompiledGraph` you can `.invoke()` or `.stream()`.

```python
app = workflow.compile()
result = app.invoke(MyState(question="How do I use LangGraph?"))
```

## State Management

State flows through every node as an immutable snapshot. Each node returns a partial dict; LangGraph merges it with the existing state to produce the next snapshot. This means nodes only need to declare the fields they change.

```python
# Node only touches `answer` — all other fields carry over unchanged
def generation_node(state: MyState) -> dict:
    return {"answer": "Here is the answer..."}
```

## Retry Patterns

A common pattern for self-corrective agents is to track retry count in the state and use a conditional edge to decide whether to retry or give up.

```python
class RAGState(BaseModel):
    question: str
    answer: str = ""
    retry_count: int = 0
    all_irrelevant: bool = False

def route_after_grading(state: RAGState) -> str:
    if not state.all_irrelevant:
        return "generate"
    if state.retry_count < 2:
        return "rewrite"
    return "give_up"
```

## Streaming

LangGraph supports streaming intermediate node outputs so your UI can show progress.

```python
for event in app.stream(initial_state):
    for node_name, output in event.items():
        print(f"Node '{node_name}' produced: {output}")
```

## Persistence and Checkpoints

LangGraph can checkpoint state at every node using a `MemorySaver` (in-memory) or a database-backed checkpointer. This enables conversation memory across sessions.

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer)

# Pass a thread_id to maintain session-level memory
config = {"configurable": {"thread_id": "session-42"}}
result = app.invoke(initial_state, config=config)
```

## Debugging

Enable verbose logging by setting `LANGCHAIN_VERBOSE=true` in your environment. Use `app.get_graph().draw_mermaid()` to visualise the graph topology as a Mermaid diagram.
