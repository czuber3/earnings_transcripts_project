from abc import ABC, abstractmethod

from mistralai import Mistral


class BaseEvaluator(ABC):
    """Abstract class for evaluators"""

    def __init__(self):
        pass
    
    @abstractmethod
    def context_answer_alignment(
        self,
        question: str,
        answer: str,
        context: str
    ):
        """Abstract method that returns True if the answer is contained in the
        given context
        
        Args:
            question (str): the question/query being asked
            answer (str): ground truth answer to the given question
            context (str): context to evaluate
        """
        pass

class LLMEvaluator(BaseEvaluator):
    """LLM evaluator. Uses Mistral model to determine if answer is contained in
    the context."""

    def __init__(self, mistral_api_key, mistral_model="mistral-large-latest"):
        self.mistral_client = Mistral(mistral_api_key)
        self.mistral_model = mistral_model

    def context_answer_alignment(
        self,
        question: str,
        answer: str,
        context: str
    )->bool:
        evaluation_prompt = f"""
            You are a helpful financial assistant. Given a query/answer pair 
            and a list of excerpts from a earnings transcript, check to see if 
            any the excerpts contain the answer to the question. Return a 1 if 
            the given answer is contained in one of the excerpts, a 0 if not. 
            Don't return any additional text.

            Question: {question}
            Answer: {answer}
            Excerpts: {context}
        """

        response = self.mistral_client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "user", "content": evaluation_prompt}
            ]
        )

        try:
            return int(response.choices[0].message.content[0])
        except:
            return 0