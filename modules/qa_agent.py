from langchain_google_genai import ChatGoogleGenerativeAI


class GeminiLLM:
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize Gemini LLM with API key (required parameter)
        """
        if not api_key:
            raise ValueError(
                "Google API key is required. Please provide api_key parameter."
            )
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.3,
            max_output_tokens=1024,
        )

    def answer_question(self, context_input: str) -> str:
        """
        Answer a question given context input
        """
        try:
            response = self.llm.invoke(context_input)
            return response.content
        except Exception as e:
            return f"Error answering question: {str(e)}"


# Keep LocalLLM for backward compatibility (deprecated)
class LocalLLM:
    def __init__(self, api_key: str):
        """
        Deprecated: Use GeminiLLM instead
        Requires API key to be passed from UI
        """
        if not api_key:
            raise ValueError("API key is required for LocalLLM (now uses Gemini)")
        print("Warning: LocalLLM is deprecated. Use GeminiLLM instead.")
        self.gemini = GeminiLLM(api_key=api_key)

    def summarize_text(self, text: str, max_chars: int = 3000) -> str:
        return self.gemini.summarize_text(text, max_chars)

    def answer_question(self, job_description: str, question: str) -> str:
        return self.gemini.answer_question(job_description, question)
