from langgraph.graph import END, StateGraph
from graph.graph_state import GraphState
from graph.graph_nodes import web_search, retrieve, grade_documents, generate, transform_query
from graph.graph_edges import route_question, decide_to_generate, grade_generation_v_documents_and_question
from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())


workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("web_search", web_search)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)

# Build graph
workflow.set_conditional_entry_point(
    route_question,
    {
        "web_search": "web_search",
        "vectorstore": "retrieve",
    },
)
workflow.add_edge("web_search", "generate")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "transform_query": "transform_query",
        "generate": "generate",
    },
)
workflow.add_edge("transform_query", "retrieve")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "transform_query",
    },
)

# Compile
app = workflow.compile()

# Run
input = {"question": "What is the AlphaCodium paper about?"}

for output in app.stream(input):
    for key, value in output.items():
        # Node
        print(f"Node: '{key}")
        # print(value["keys"], indent=2, width=80, depth=None)
    
    print("\n---\n")
    
# Final generation
print(value["generation"])    
