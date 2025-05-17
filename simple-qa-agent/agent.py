import os
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def generate_completion(self, messages, max_tokens=500):
        """Generate a completion from the LLM"""
        pass

    @abstractmethod
    def get_available_models(self):
        """Get a list of available models from this provider"""
        pass

    @classmethod
    def is_valid_model(cls, model_name):
        """Check if a model name is valid for this provider"""
        return model_name in cls().get_available_models()


class OpenAIProvider(LLMProvider):
    """OpenAI API provider implementation"""

    def __init__(self, api_key=None):
        try:
            from openai import OpenAI
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")

    def generate_completion(self, messages, max_tokens=500):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content, None
        except Exception as e:
            error_message = str(e).lower()

            if any(keyword in error_message for keyword in ["authentication", "auth", "key"]):
                return None, "Authentication failed. Please check your API key."
            elif any(keyword in error_message for keyword in ["model", "not found", "doesn't exist"]):
                return None, f"The model '{self.model_name}' is not available. Please try a different model."
            else:
                return None, f"Error: {str(e)}"

    def get_available_models(self):
        try:
            models = self.client.models.list()
            return [model.id for model in models]
        except Exception:
            # Fallback to common models if API call fails
            return [
                "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4",
                "gpt-3.5-turbo", "gpt-3.5-turbo-16k"
            ]

    def set_model(self, model_name):
        self.model_name = model_name


class AnthropicProvider(LLMProvider):
    """Anthropic API provider implementation"""

    def __init__(self, api_key=None):
        try:
            import anthropic
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.model_name = "claude-3-opus-20240229"  # Default model
        except ImportError:
            raise ImportError("Anthropic package not installed. Install with: pip install anthropic")

    def generate_completion(self, messages, max_tokens=500):
        try:
            # Convert chat format to Anthropic format
            prompt = self._convert_messages_to_anthropic_format(messages)

            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens,
                messages=prompt
            )
            return response.content[0].text, None
        except Exception as e:
            return None, f"Anthropic API error: {str(e)}"

    def _convert_messages_to_anthropic_format(self, messages):
        """Convert OpenAI-style messages to Anthropic format"""
        global system_message
        anthropic_messages = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                # System messages handled differently in Anthropic
                system_message = content
                continue

            if role == "user":
                anthropic_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                anthropic_messages.append({"role": "assistant", "content": content})

        # Add system message if it exists
        if system_message:
            # In newer Anthropic API, system goes as a parameter
            anthropic_messages.insert(0, {"role": "system", "content": system_message})

        return anthropic_messages

    def get_available_models(self):
        # Anthropic doesn't have a list models endpoint, so we hardcode common models
        return [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2"
        ]

    def set_model(self, model_name):
        self.model_name = model_name


class CodingAssistant:
    """A generic coding assistant that can use different LLM providers"""

    def __init__(self, provider_name="openai", model_name=None, api_key=None):
        # Initialize the provider
        self.provider = self._get_provider(provider_name, api_key)

        # Set default model if none provided
        if model_name is None:
            if provider_name == "openai":
                model_name = "gpt-3.5-turbo"
            elif provider_name == "anthropic":
                model_name = "claude-3-haiku-20240307"

        # Set the model
        self.provider.set_model(model_name)

        # Initialize conversation history
        self.conversation_history = []

        # Set a system prompt focused on coding assistance
        self.system_prompt = """
        You are a specialized coding assistant focused exclusively on programming-related queries.
        
        Guidelines:
        1. Only respond to questions related to programming, software development, algorithms, 
           data structures, debugging, or development tools.
        2. For non-coding questions, politely explain that you're a specialized coding assistant 
           and can only help with programming-related topics.
        3. Provide clear, well-commented code examples when appropriate.
        4. Explain your code and reasoning to help the user learn.
        5. If you're unsure about something, acknowledge the limitations rather than guessing.
        6. Focus on best practices and secure coding standards.
        
        Remember, your purpose is to help users become better programmers through 
        accurate, educational, and helpful responses to coding questions.
        """

    def _get_provider(self, provider_name, api_key=None):
        """Get the appropriate LLM provider based on name"""
        if provider_name.lower() == "openai":
            return OpenAIProvider(api_key)
        elif provider_name.lower() == "anthropic":
            return AnthropicProvider(api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider_name}")

    def add_to_history(self, role, content):
        """Add a message to the conversation history"""
        self.conversation_history.append({"role": role, "content": content})

    def is_coding_related(self, query):
        """Determine if a query is related to coding/programming"""
        # Keywords that suggest coding-related queries
        coding_keywords = [
            "code", "program", "function", "class", "method", "variable",
            "algorithm", "data structure", "api", "framework", "library",
            "debugging", "error", "exception", "syntax", "compiler",
            "interpreter", "runtime", "development", "software", "git",
            "html", "css", "javascript", "python", "java", "c++", "c#",
            "ruby", "php", "sql", "database", "frontend", "backend",
            "fullstack", "web", "mobile", "app", "development", "devops",
            "cloud", "server", "client", "api", "rest", "json", "xml",
            "http", "request", "response", "async", "promise", "callback",
            "bug", "fix", "issue", "implement", "feature", "test", "unit test",
            "integration test", "deployment", "build", "package", "module",
            "import", "export", "dependency", "npm", "pip", "gem", "nuget",
            "docker", "kubernetes", "container", "virtual machine", "vm",
            "ide", "editor", "terminal", "command line", "shell", "bash",
            "powershell", "script", "automation", "ci/cd", "continuous integration",
            "version control", "repository", "commit", "merge", "pull request",
            "branch", "checkout", "clone", "fork", "open source", "license",
            "ownership",
        ]

        # Check if any coding keyword is in the query
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in coding_keywords)

    def get_response(self, user_input):
        """Process user input and generate a response"""
        # Check if the query is coding-related
        if not self.is_coding_related(user_input):
            return ("I'm a specialized coding assistant and can only help with programming-related questions. "
                    "Please ask me about coding, software development, algorithms, debugging, or related topics.")

        # Add user input to history
        self.add_to_history("user", user_input)

        # Prepare messages for the API call
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)

        # Get response from the provider
        response_text, error = self.provider.generate_completion(messages)

        if error:
            return f"Error: {error}"

        # Add response to history
        self.add_to_history("assistant", response_text)

        return response_text

    def change_provider(self, provider_name, model_name=None, api_key=None):
        """Change the LLM provider"""
        try:
            new_provider = self._get_provider(provider_name, api_key)

            # Set default model if none provided
            if model_name is None:
                if provider_name == "openai":
                    model_name = "gpt-3.5-turbo"
                elif provider_name == "anthropic":
                    model_name = "claude-3-haiku-20240307"

            new_provider.set_model(model_name)

            # If successful, update the provider
            self.provider = new_provider
            return f"Provider changed to {provider_name} using model {model_name}"
        except Exception as e:
            return f"Error changing provider: {str(e)}"

    def change_model(self, model_name):
        """Change the model being used by the current provider"""
        try:
            self.provider.set_model(model_name)
            return f"Model changed to {model_name}"
        except Exception as e:
            return f"Error changing model: {str(e)}"

    def clear_history(self):
        """Clear the conversation history"""
        self.conversation_history = []
        return "Conversation history cleared."