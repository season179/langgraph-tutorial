from router import question_router
from hallucination_grader import hallucination_grader
from answer_grader import answer_grader

def route_question(state):
    """
    Route question to either web search or RAG
    
    Args:
        state (dict): The current state of the graph
        
    Returns:
        str: Next node to call
    """
    
    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    source = question_router.invoke({"question": question})
    data_source = source["datasource"]
    print(data_source)
    
    if data_source == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "web_search"
    elif data_source == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    
    
def decide_to_generate(state):
    """
    Decide whether to generate an answer, or re-generate a new question
    
    Args:
        state (dict): The current state of the graph
        
    Returns:
        str: Next node to call
    """
    
    print("---ASSESS GRADED DOCUMENTS---")
    filtered_documents = state["documents"]
    
    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new question
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, REGENERATE QUESTION---")
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE ANSWER---")
        return "generate"
    
    
def grade_generation_v_documents_and_question(state):
    """
    Decide whether the generation is grounded in the document and answer the question.
    
    Args:
        state (dict): The current state of the graph
        
    Returns:
        str: Next node to call
    """
    
    print("---CHECK HALLUCINATION---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    
    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score["score"]
    
    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        
        # Check question answering
        print("---GRADE GENERATION VS QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ANSWER QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
