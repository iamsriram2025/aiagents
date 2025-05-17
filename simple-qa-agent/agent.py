import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SimpleQAAgent:

    def __init__(self, model_name="gpt-3.5-turbo"):
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        self.conversation_history = []
        self.system_prompt = """
        You are a helpful assistant that provides clear, concise, and accurate answers.
        Respond in a friendly and informative manner.
        If you don't know the answer, admit it rather than making up information.
        """

    def add_to_history(self, role, content):
        """Add a message to the conversation history."""
        self.conversation_history.append({"role": role, "content": content})

    def get_response(self, user_input):
        """Process user input and generate a response."""
        # Add user input to history
        self.add_to_history("user", user_input)

        # Prepare messages for the API call
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)

        # Call the OpenAI API with generic error handling
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=500
            )

            # Extract the response text
            response_text = response.choices[0].message.content

            # Add response to history
            self.add_to_history("assistant", response_text)

            return response_text

        except Exception as e:
            error_message = str(e).lower()

            # Parse the error message to determine the type of error
            if any(keyword in error_message for keyword in ["authentication", "auth", "key", "invalid key"]):
                return "Error: Authentication failed. Please check your API key."

            elif any(keyword in error_message for keyword in ["rate limit", "ratelimit", "requests", "quota"]):
                return "Error: Rate limit exceeded. Please try again later."

            elif any(keyword in error_message for keyword in ["model", "not found", "doesn't exist", "does not exist"]):
                return f"Error: The model '{self.model_name}' is not available. Please try a different model."

            elif any(keyword in error_message for keyword in ["connection", "network", "timeout", "connect"]):
                return "Error: Could not connect to the OpenAI API. Please check your internet connection."

            else:
                # Generic error message with the original error
                return f"Error occurred: {str(e)}"

    def set_model(self, model_name):
        """Change the model being used by the agent."""
        self.model_name = model_name
        return f"Model changed to {model_name}"

    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []