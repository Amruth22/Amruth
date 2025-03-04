"""
Evaluation and feedback loop system for RAG
"""

import json
import datetime
from typing import Dict, List, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Initialize models
llm = ChatOpenAI(model="gpt-4o", temperature=0)

class RAGEvaluator:
    """
    Evaluator for RAG system responses
    """
    
    def __init__(self, log_file: str = "rag_evaluation_logs.jsonl"):
        """
        Initialize the RAG evaluator
        
        Args:
            log_file: Path to the log file for storing evaluation results
        """
        self.log_file = log_file
        self.evaluation_metrics = [
            "relevance",
            "faithfulness",
            "context_utilization",
            "answer_completeness",
            "hallucination"
        ]
    
    def evaluate_response(
        self, 
        query: str, 
        response: str, 
        retrieved_documents: List[Document],
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a RAG response based on multiple criteria
        
        Args:
            query: The user's query
            response: The generated response
            retrieved_documents: The documents retrieved for the query
            ground_truth: Optional ground truth answer for comparison
            
        Returns:
            Dictionary containing evaluation scores and feedback
        """
        # Format context for evaluation
        context = "\n\n".join([
            f"Document {i+1}:\nSource: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}"
            for i, doc in enumerate(retrieved_documents)
        ])
        
        # Create evaluation prompt
        evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert evaluator for Retrieval-Augmented Generation (RAG) systems.
            Your task is to evaluate the quality of a RAG system's response based on the following criteria:
            
            1. Relevance (1-10): How relevant is the response to the query?
            2. Faithfulness (1-10): How faithful is the response to the retrieved context?
            3. Context Utilization (1-10): How well does the response utilize the retrieved context?
            4. Answer Completeness (1-10): How complete is the answer?
            5. Hallucination (1-10, higher is better): Is the response free from hallucinations or made-up information?
            
            Provide a score for each criterion and a brief explanation for your rating.
            Also provide specific feedback on how the response could be improved.
            
            Format your response as a JSON object with the following structure:
            {
                "relevance": {"score": X, "explanation": "..."},
                "faithfulness": {"score": X, "explanation": "..."},
                "context_utilization": {"score": X, "explanation": "..."},
                "answer_completeness": {"score": X, "explanation": "..."},
                "hallucination": {"score": X, "explanation": "..."},
                "overall_score": X,
                "improvement_feedback": "..."
            }
            """),
            ("human", """Query: {query}
            
            Retrieved Context:
            {context}
            
            Response:
            {response}
            
            {ground_truth_prompt}
            
            Evaluate the response based on the criteria described.
            """)
        ])
        
        # Add ground truth if available
        ground_truth_prompt = ""
        if ground_truth:
            ground_truth_prompt = f"Ground Truth Answer:\n{ground_truth}"
        
        # Generate evaluation
        evaluation_input = {
            "query": query,
            "context": context,
            "response": response,
            "ground_truth_prompt": ground_truth_prompt
        }
        
        evaluation_result = llm.invoke(evaluation_prompt.format_messages(**evaluation_input))
        
        try:
            # Parse the evaluation result
            evaluation_data = json.loads(evaluation_result.content)
            
            # Calculate overall score if not provided
            if "overall_score" not in evaluation_data:
                scores = [
                    evaluation_data[metric]["score"] 
                    for metric in self.evaluation_metrics
                    if metric in evaluation_data
                ]
                evaluation_data["overall_score"] = sum(scores) / len(scores)
            
            # Log the evaluation
            self._log_evaluation(
                query=query,
                response=response,
                context=context,
                ground_truth=ground_truth,
                evaluation=evaluation_data
            )
            
            return evaluation_data
        
        except json.JSONDecodeError:
            # Fallback for parsing errors
            return {
                "error": "Failed to parse evaluation result",
                "raw_evaluation": evaluation_result.content
            }
    
    def _log_evaluation(
        self,
        query: str,
        response: str,
        context: str,
        ground_truth: Optional[str],
        evaluation: Dict[str, Any]
    ) -> None:
        """
        Log the evaluation results to a file
        
        Args:
            query: The user's query
            response: The generated response
            context: The retrieved context
            ground_truth: The ground truth answer (if available)
            evaluation: The evaluation results
        """
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "response": response,
            "context_length": len(context),
            "ground_truth": ground_truth,
            "evaluation": evaluation
        }
        
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def analyze_evaluation_logs(self, n_entries: int = 100) -> Dict[str, Any]:
        """
        Analyze evaluation logs to identify patterns and areas for improvement
        
        Args:
            n_entries: Number of most recent log entries to analyze
            
        Returns:
            Analysis results
        """
        try:
            # Read log entries
            log_entries = []
            with open(self.log_file, "r") as f:
                for line in f:
                    log_entries.append(json.loads(line.strip()))
            
            # Take the most recent n entries
            log_entries = log_entries[-n_entries:]
            
            if not log_entries:
                return {"error": "No log entries found"}
            
            # Calculate average scores
            metric_scores = {metric: [] for metric in self.evaluation_metrics}
            overall_scores = []
            
            for entry in log_entries:
                evaluation = entry.get("evaluation", {})
                
                # Skip entries with parsing errors
                if "error" in evaluation:
                    continue
                
                for metric in self.evaluation_metrics:
                    if metric in evaluation and "score" in evaluation[metric]:
                        metric_scores[metric].append(evaluation[metric]["score"])
                
                if "overall_score" in evaluation:
                    overall_scores.append(evaluation["overall_score"])
            
            # Calculate averages
            avg_scores = {
                metric: sum(scores) / len(scores) if scores else 0
                for metric, scores in metric_scores.items()
            }
            
            avg_overall = sum(overall_scores) / len(overall_scores) if overall_scores else 0
            
            # Identify weakest areas
            weakest_metrics = sorted(
                [(metric, avg) for metric, avg in avg_scores.items()],
                key=lambda x: x[1]
            )
            
            # Generate improvement suggestions
            improvement_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert at analyzing RAG system performance.
                Based on the evaluation metrics provided, suggest specific improvements
                to enhance the system's performance in the weakest areas."""),
                ("human", """Evaluation metrics:
                {metrics}
                
                Overall average score: {overall_avg}/10
                
                Weakest areas:
                {weakest_areas}
                
                Suggest specific technical improvements to address these weaknesses.
                Focus on practical changes to the RAG system architecture, retrieval mechanism,
                prompt engineering, or other components.""")
            ])
            
            improvement_input = {
                "metrics": json.dumps(avg_scores, indent=2),
                "overall_avg": round(avg_overall, 2),
                "weakest_areas": "\n".join([f"{metric}: {round(score, 2)}/10" for metric, score in weakest_metrics[:3]])
            }
            
            improvement_suggestions = llm.invoke(
                improvement_prompt.format_messages(**improvement_input)
            ).content
            
            # Return analysis results
            return {
                "num_entries_analyzed": len(log_entries),
                "average_scores": avg_scores,
                "overall_average": avg_overall,
                "weakest_areas": [{"metric": metric, "score": score} for metric, score in weakest_metrics[:3]],
                "improvement_suggestions": improvement_suggestions
            }
        
        except Exception as e:
            return {"error": f"Failed to analyze logs: {str(e)}"}
    
    def generate_feedback_for_user(
        self,
        query: str,
        response: str,
        evaluation: Dict[str, Any]
    ) -> str:
        """
        Generate user-friendly feedback based on evaluation results
        
        Args:
            query: The user's query
            response: The generated response
            evaluation: The evaluation results
            
        Returns:
            User-friendly feedback
        """
        feedback_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an assistant that provides helpful feedback on answers.
            Based on the evaluation of a response, generate friendly and constructive feedback
            for the user about the quality of the answer they received.
            
            Keep the feedback concise, helpful, and focused on the most important aspects.
            Don't mention the specific evaluation scores, but do highlight strengths and
            potential limitations of the answer."""),
            ("human", """Query: {query}
            
            Response: {response}
            
            Evaluation: {evaluation}
            
            Generate user-friendly feedback about this response:""")
        ])
        
        feedback_input = {
            "query": query,
            "response": response,
            "evaluation": json.dumps(evaluation, indent=2)
        }
        
        feedback = llm.invoke(
            feedback_prompt.format_messages(**feedback_input)
        ).content
        
        return feedback

class FeedbackLoop:
    """
    Feedback loop for improving RAG system based on evaluations
    """
    
    def __init__(self, vector_store):
        """
        Initialize the feedback loop
        
        Args:
            vector_store: The vector store to update
        """
        self.vector_store = vector_store
        self.evaluator = RAGEvaluator()
        self.feedback_log_file = "rag_feedback_logs.jsonl"
    
    def process_user_feedback(
        self,
        query: str,
        response: str,
        retrieved_documents: List[Document],
        user_rating: int,
        user_feedback: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process user feedback and update the system
        
        Args:
            query: The user's query
            response: The generated response
            retrieved_documents: The documents retrieved for the query
            user_rating: User rating (1-5)
            user_feedback: Optional user feedback text
            
        Returns:
            Processing results
        """
        # Log the feedback
        feedback_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "query": query,
            "response": response,
            "user_rating": user_rating,
            "user_feedback": user_feedback
        }
        
        with open(self.feedback_log_file, "a") as f:
            f.write(json.dumps(feedback_entry) + "\n")
        
        # For low ratings, perform automatic evaluation
        if user_rating <= 3:
            evaluation = self.evaluator.evaluate_response(
                query=query,
                response=response,
                retrieved_documents=retrieved_documents
            )
            
            # Generate improved query if needed
            if evaluation.get("overall_score", 0) < 7:
                improved_query = self._generate_improved_query(
                    query=query,
                    response=response,
                    evaluation=evaluation,
                    user_feedback=user_feedback
                )
                
                return {
                    "evaluation": evaluation,
                    "improved_query": improved_query,
                    "action": "query_improvement"
                }
        
        return {
            "action": "feedback_logged",
            "message": "Thank you for your feedback!"
        }
    
    def _generate_improved_query(
        self,
        query: str,
        response: str,
        evaluation: Dict[str, Any],
        user_feedback: Optional[str]
    ) -> str:
        """
        Generate an improved query based on evaluation and feedback
        
        Args:
            query: The original query
            response: The generated response
            evaluation: The evaluation results
            user_feedback: User feedback text
            
        Returns:
            Improved query
        """
        query_improvement_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at improving search queries.
            Based on the original query, the response, evaluation, and user feedback,
            generate an improved query that would lead to better search results.
            
            The improved query should:
            1. Be more specific and targeted
            2. Include key terms that might improve retrieval
            3. Address any issues identified in the evaluation or feedback"""),
            ("human", """Original Query: {query}
            
            Response: {response}
            
            Evaluation: {evaluation}
            
            User Feedback: {user_feedback}
            
            Generate an improved search query:""")
        ])
        
        query_input = {
            "query": query,
            "response": response,
            "evaluation": json.dumps(evaluation, indent=2),
            "user_feedback": user_feedback or "No specific feedback provided"
        }
        
        improved_query = llm.invoke(
            query_improvement_prompt.format_messages(**query_input)
        ).content
        
        return improved_query
    
    def update_retrieval_strategy(self) -> Dict[str, Any]:
        """
        Analyze feedback and evaluation logs to update retrieval strategy
        
        Returns:
            Update results
        """
        # Analyze evaluation logs
        analysis = self.evaluator.analyze_evaluation_logs()
        
        if "error" in analysis:
            return {"error": analysis["error"]}
        
        # Generate retrieval strategy improvements
        strategy_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at optimizing RAG retrieval strategies.
            Based on the analysis of evaluation logs, suggest specific changes to
            the retrieval strategy to improve performance."""),
            ("human", """Evaluation Analysis:
            {analysis}
            
            Suggest specific changes to the retrieval strategy:""")
        ])
        
        strategy_input = {
            "analysis": json.dumps(analysis, indent=2)
        }
        
        strategy_suggestions = llm.invoke(
            strategy_prompt.format_messages(**strategy_input)
        ).content
        
        return {
            "analysis": analysis,
            "strategy_suggestions": strategy_suggestions
        }