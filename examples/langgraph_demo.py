"""Minimal LangGraph contrast; install with `pip install -e '.[graph]'`."""
from typing import TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    messages: list[str]


def researcher(state: State) -> State:
    return {"messages": state["messages"] + ["research complete"]}


def verifier(state: State) -> State:
    return {"messages": state["messages"] + ["verified"]}


graph = StateGraph(State)
graph.add_node("researcher", researcher)
graph.add_node("verifier", verifier)
graph.add_edge(START, "researcher")
graph.add_edge("researcher", "verifier")
graph.add_edge("verifier", END)
app = graph.compile()

if __name__ == "__main__":
    print(app.invoke({"messages": ["start"]}))

