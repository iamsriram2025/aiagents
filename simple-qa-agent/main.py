from agent import CodingAssistant


def main():
    print("🤖 Multi-LLM Coding Assistant 🤖")
    print("This assistant is specialized for programming and coding questions only.")
    print("\nCommands:")
    print("- 'exit': Quit the application")
    print("- 'clear': Reset conversation history")
    print("- 'model <name>': Change the LLM model (e.g., 'model gpt-4')")
    print("- 'provider <name> [model] [api_key]': Change the LLM provider")
    print("  Example: provider anthropic claude-3-haiku-20240307 your_api_key_here")
    print("-" * 70)

    # Initialize the assistant with default provider and model
    assistant = CodingAssistant()

    # Main interaction loop
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        # Check for exit command
        if user_input.lower() == "exit":
            print("Goodbye! Happy coding!")
            break

        # Check for clear command
        elif user_input.lower() == "clear":
            result = assistant.clear_history()
            print(f"\nSystem: {result}")
            continue

        # Check for model change command
        elif user_input.lower().startswith("model "):
            try:
                # Extract model name
                new_model = user_input[6:].strip()
                if new_model:
                    result = assistant.change_model(new_model)
                    print(f"\nSystem: {result}")
                else:
                    print("\nSystem: Please specify a model name.")
            except Exception as e:
                print(f"\nSystem Error: {str(e)}")
            continue

        # Check for provider change command
        elif user_input.lower().startswith("provider "):
            try:
                # Parse provider command: provider <name> [model] [api_key]
                parts = user_input[9:].strip().split()
                provider_name = parts[0] if parts else ""
                model_name = parts[1] if len(parts) > 1 else None
                api_key = parts[2] if len(parts) > 2 else None

                if provider_name:
                    result = assistant.change_provider(provider_name, model_name, api_key)
                    print(f"\nSystem: {result}")
                else:
                    print("\nSystem: Please specify a provider name.")
            except Exception as e:
                print(f"\nSystem Error: {str(e)}")
            continue

        # Get response from assistant for regular queries
        elif user_input:
            print("\nThinking...")
            response = assistant.get_response(user_input)

            # Check if response is an error message
            if response.startswith("Error:"):
                print(f"\nSystem: {response}")
            else:
                print(f"\nAssistant: {response}")

if __name__ == "__main__":
    main()