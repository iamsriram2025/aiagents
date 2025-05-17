from agent import SimpleQAAgent

def main():
    print("Simple Q&A Agent")
    print("Commands:")
    print("- 'exit': Quit the application")
    print("- 'clear': Reset conversation history")
    print("- 'model <name>': Change the LLM model (e.g., 'model gpt-4')")
    print("-" * 50)

    # Initialize the agent with default model
    agent = SimpleQAAgent()

    # Main interaction loop
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        # Check for exit command
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # Check for clear command
        elif user_input.lower() == "clear":
            agent.clear_history()
            print("Conversation history cleared.")
            continue

        # Check for model change command
        elif user_input.lower().startswith("model "):
            try:
                # Extract model name
                new_model = user_input[6:].strip()
                if new_model:
                    result = agent.set_model(new_model)
                    print(f"\nSystem: {result}")
                else:
                    print("\nSystem: Please specify a model name.")
            except Exception as e:
                print(f"\nSystem Error: {str(e)}")
            continue

        # Get response from agent
        elif user_input:
            print("\nThinking...")
            response = agent.get_response(user_input)

            # Check if response is an error message
            if response.startswith("Error:") or response.startswith("API Error:") or response.startswith("Unexpected error:"):
                print(f"\nSystem: {response}")
            else:
                print(f"\nAssistant: {response}")

if __name__ == "__main__":
    main()