from langgraph.graph import StateGraph
from agent.tools import analyze_student, web_search
from agent.llm import generate_response
from agent.rag import retrieve


def agent_node(state):
    try:
        print("STATE RECEIVED:", state)

        user_input = state.get("input", "").lower()

        if not user_input:
            return {"response": "Please enter something."}

        # 🔹 ANALYSIS
        if "analyze" in user_input:
            if "file" not in state:
                return {"response": "Please upload/provide CSV file path first."}

            try:
                result = analyze_student(state["file"])
                return {"analysis": result}
            except Exception as e:
                return {"response": f"Analysis Error: {str(e)}"}

        # 🔹 PLAN GENERATION
        elif "plan" in user_input:
            context = state.get("analysis")

            if not context:
                return {"response": "Please run 'analyze student' first."}

            prompt = f"""
            You are an AI Study Coach.

            Student Summary:
            {context}

            Give:
            1. Weakness Analysis (short)
            2. Weekly Study Plan (max 5 points)
            3. Tips (3 bullet points)

            Keep it concise and structured.
            """

            try:
                return {"plan": generate_response(prompt)}
            except Exception as e:
                return {"response": f"Plan Generation Error: {str(e)}"}

        # 🔹 RESOURCE RECOMMENDATION
        elif "resource" in user_input:
            try:
                docs = retrieve(user_input)
                links = web_search(user_input)

                prompt = f"""
                Query: {user_input}

                Context:
                {docs}

                Links:
                {links}

                Suggest best learning resources clearly.
                """

                return {
                    "resources": generate_response(prompt),
                    "links": links
                }

            except Exception as e:
                return {"response": f"Resource Error: {str(e)}"}

        # 🔹 DEFAULT CHAT
        return {"response": generate_response(user_input)}

    except Exception as e:
        return {"response": f"Agent Error: {str(e)}"}


# Build graph safely
try:
    graph = StateGraph(dict)

    graph.add_node("agent", agent_node)
    graph.set_entry_point("agent")

    app = graph.compile()
    print("✅ Graph compiled successfully")

except Exception as e:
    print("❌ Graph build failed:", e)
    app = None