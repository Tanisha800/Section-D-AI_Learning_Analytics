from agent.graph import app

state = {}

print("📚 AI Study Coach (type 'exit' to quit)")

while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "exit":
        break

    state["input"] = user_input

    if "analyze" in user_input.lower():
        file_path = input("Enter CSV path: ")
        state["file"] = file_path

    result = app.invoke(state)

    if result:
        state.update(result)

    if "analysis" in result:
        print("\n📊 Analysis Summary:")
        print(result["analysis"])

    elif "plan" in result:
        print("\n📅 Study Plan:\n")
        print(result["plan"])

    elif "resources" in result:
        print("\n📚 Resources:\n")
        print(result["resources"])
        print("\n🔗 Links:")
        for link in result["links"]:
            print("-", link)

    elif "response" in result:
        print("\nAI:", result["response"])