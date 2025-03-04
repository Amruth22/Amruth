"""
Structured output and tool use capabilities for the RAG system
"""

import json
import datetime
import requests
from typing import Dict, List, Any, Optional, Callable, Union
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Initialize models
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Define structured output schemas
class SearchResult(BaseModel):
    """Search result from the RAG system"""
    answer: str = Field(description="The answer to the user's question")
    sources: List[str] = Field(description="List of sources used to generate the answer")
    confidence: float = Field(description="Confidence score between 0 and 1")
    related_questions: List[str] = Field(description="Related questions the user might ask next")

class AnalysisResult(BaseModel):
    """Analysis result from the RAG system"""
    summary: str = Field(description="Summary of the analysis")
    key_points: List[str] = Field(description="Key points extracted from the analysis")
    entities: Dict[str, List[str]] = Field(description="Entities mentioned in the content")
    sentiment: str = Field(description="Overall sentiment of the content")
    recommendations: List[str] = Field(description="Recommendations based on the analysis")

class ComparisonResult(BaseModel):
    """Comparison result from the RAG system"""
    similarities: List[str] = Field(description="Similarities between the compared items")
    differences: List[str] = Field(description="Differences between the compared items")
    evaluation: Dict[str, Any] = Field(description="Evaluation metrics for the comparison")
    recommendation: str = Field(description="Recommendation based on the comparison")

# Define tools
@tool
def search_web(query: str) -> str:
    """
    Search the web for information about a query.
    
    Args:
        query: The search query
        
    Returns:
        Search results as text
    """
    # This is a mock implementation - in production, integrate with a real search API
    return f"Web search results for: {query}\n\n" + \
           f"1. First result about {query}\n" + \
           f"2. Second result about {query}\n" + \
           f"3. Third result about {query}"

@tool
def get_weather(location: str) -> str:
    """
    Get current weather information for a location.
    
    Args:
        location: The location to get weather for
        
    Returns:
        Weather information
    """
    # This is a mock implementation - in production, integrate with a weather API
    return f"Weather for {location}: 72°F, Partly Cloudy"

@tool
def calculate(expression: str) -> str:
    """
    Calculate the result of a mathematical expression.
    
    Args:
        expression: The mathematical expression to calculate
        
    Returns:
        The result of the calculation
    """
    try:
        # Warning: eval can be dangerous in production
        # Use a safer alternative in real applications
        result = eval(expression)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"

# Tool registry
TOOLS = {
    "search_web": search_web,
    "get_weather": get_weather,
    "calculate": calculate
}

class StructuredOutputGenerator:
    """
    Generator for structured outputs from the RAG system
    """
    
    def __init__(self):
        """Initialize the structured output generator"""
        self.llm = llm
    
    def generate_search_result(
        self,
        query: str,
        context: List[str],
        metadata: List[Dict[str, Any]]
    ) -> SearchResult:
        """
        Generate a structured search result
        
        Args:
            query: The user's query
            context: List of context passages
            metadata: Metadata for each context passage
            
        Returns:
            Structured search result
        """
        # Create the prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an assistant that generates structured search results.
            Based on the user's query and the provided context, generate a comprehensive answer.
            
            Your response should be formatted as a JSON object matching the SearchResult schema:
            {
                "answer": "The detailed answer to the user's question",
                "sources": ["source1", "source2", ...],
                "confidence": 0.95,
                "related_questions": ["related question 1", "related question 2", ...]
            }
            
            Ensure your answer is accurate, comprehensive, and based only on the provided context.
            The confidence score should reflect how well the context supports the answer.
            """),
            ("human", """Query: {query}
            
            Context:
            {context}
            
            Generate a structured search result:""")
        ])
        
        # Format the context
        formatted_context = "\n\n".join([
            f"Source: {metadata[i].get('source', 'Unknown')}\n{context[i]}"
            for i in range(len(context))
        ])
        
        # Generate the response
        response = self.llm.invoke(
            prompt.format_messages(
                query=query,
                context=formatted_context
            )
        )
        
        # Parse the response
        try:
            result_dict = json.loads(response.content)
            return SearchResult(**result_dict)
        except Exception as e:
            # Fallback for parsing errors
            return SearchResult(
                answer=f"Error generating structured result: {str(e)}",
                sources=[],
                confidence=0.0,
                related_questions=[]
            )
    
    def generate_analysis_result(
        self,
        query: str,
        context: List[str],
        metadata: List[Dict[str, Any]]
    ) -> AnalysisResult:
        """
        Generate a structured analysis result
        
        Args:
            query: The user's query
            context: List of context passages
            metadata: Metadata for each context passage
            
        Returns:
            Structured analysis result
        """
        # Create the prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an assistant that generates structured analysis results.
            Based on the user's query and the provided context, generate a comprehensive analysis.
            
            Your response should be formatted as a JSON object matching the AnalysisResult schema:
            {
                "summary": "Summary of the analysis",
                "key_points": ["key point 1", "key point 2", ...],
                "entities": {
                    "people": ["person1", "person2", ...],
                    "organizations": ["org1", "org2", ...],
                    "locations": ["location1", "location2", ...],
                    "concepts": ["concept1", "concept2", ...]
                },
                "sentiment": "positive/negative/neutral",
                "recommendations": ["recommendation1", "recommendation2", ...]
            }
            
            Ensure your analysis is accurate, comprehensive, and based only on the provided context.
            """),
            ("human", """Query: {query}
            
            Context:
            {context}
            
            Generate a structured analysis result:""")
        ])
        
        # Format the context
        formatted_context = "\n\n".join([
            f"Source: {metadata[i].get('source', 'Unknown')}\n{context[i]}"
            for i in range(len(context))
        ])
        
        # Generate the response
        response = self.llm.invoke(
            prompt.format_messages(
                query=query,
                context=formatted_context
            )
        )
        
        # Parse the response
        try:
            result_dict = json.loads(response.content)
            return AnalysisResult(**result_dict)
        except Exception as e:
            # Fallback for parsing errors
            return AnalysisResult(
                summary=f"Error generating structured analysis: {str(e)}",
                key_points=[],
                entities={},
                sentiment="neutral",
                recommendations=[]
            )
    
    def generate_comparison_result(
        self,
        query: str,
        items_to_compare: List[str],
        context: List[str],
        metadata: List[Dict[str, Any]]
    ) -> ComparisonResult:
        """
        Generate a structured comparison result
        
        Args:
            query: The user's query
            items_to_compare: List of items to compare
            context: List of context passages
            metadata: Metadata for each context passage
            
        Returns:
            Structured comparison result
        """
        # Create the prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an assistant that generates structured comparison results.
            Based on the user's query, the items to compare, and the provided context,
            generate a comprehensive comparison.
            
            Your response should be formatted as a JSON object matching the ComparisonResult schema:
            {
                "similarities": ["similarity1", "similarity2", ...],
                "differences": ["difference1", "difference2", ...],
                "evaluation": {
                    "criterion1": "evaluation1",
                    "criterion2": "evaluation2",
                    ...
                },
                "recommendation": "Overall recommendation based on the comparison"
            }
            
            Ensure your comparison is accurate, comprehensive, and based only on the provided context.
            """),
            ("human", """Query: {query}
            
            Items to Compare:
            {items_to_compare}
            
            Context:
            {context}
            
            Generate a structured comparison result:""")
        ])
        
        # Format the context
        formatted_context = "\n\n".join([
            f"Source: {metadata[i].get('source', 'Unknown')}\n{context[i]}"
            for i in range(len(context))
        ])
        
        # Format items to compare
        formatted_items = "\n".join([
            f"{i+1}. {item}" for i, item in enumerate(items_to_compare)
        ])
        
        # Generate the response
        response = self.llm.invoke(
            prompt.format_messages(
                query=query,
                items_to_compare=formatted_items,
                context=formatted_context
            )
        )
        
        # Parse the response
        try:
            result_dict = json.loads(response.content)
            return ComparisonResult(**result_dict)
        except Exception as e:
            # Fallback for parsing errors
            return ComparisonResult(
                similarities=[],
                differences=[],
                evaluation={},
                recommendation=f"Error generating structured comparison: {str(e)}"
            )

class ToolUseManager:
    """
    Manager for tool use in the RAG system
    """
    
    def __init__(self):
        """Initialize the tool use manager"""
        self.llm = llm
        self.tools = TOOLS
    
    def detect_tool_needs(self, query: str) -> Dict[str, Any]:
        """
        Detect if the query requires the use of tools
        
        Args:
            query: The user's query
            
        Returns:
            Dictionary with tool detection results
        """
        # Create the prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an assistant that analyzes queries to determine if they require external tools.
            You have access to the following tools:
            
            1. search_web: Search the web for information
            2. get_weather: Get current weather information for a location
            3. calculate: Calculate the result of a mathematical expression
            
            Analyze the query and determine if any of these tools are needed to provide a complete answer.
            
            Your response should be formatted as a JSON object:
            {
                "needs_tool": true/false,
                "tool_name": "name of the tool needed or null if none needed",
                "tool_input": "input to pass to the tool or null if none needed",
                "reasoning": "explanation of why the tool is needed or not needed"
            }
            """),
            ("human", """Query: {query}
            
            Analyze if this query requires the use of tools:""")
        ])
        
        # Generate the response
        response = self.llm.invoke(
            prompt.format_messages(query=query)
        )
        
        # Parse the response
        try:
            result = json.loads(response.content)
            return result
        except Exception as e:
            # Fallback for parsing errors
            return {
                "needs_tool": False,
                "tool_name": None,
                "tool_input": None,
                "reasoning": f"Error detecting tool needs: {str(e)}"
            }
    
    def execute_tool(self, tool_name: str, tool_input: str) -> str:
        """
        Execute a tool with the given input
        
        Args:
            tool_name: Name of the tool to execute
            tool_input: Input to pass to the tool
            
        Returns:
            Tool execution result
        """
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found"
        
        try:
            tool_fn = self.tools[tool_name]
            result = tool_fn(tool_input)
            return result
        except Exception as e:
            return f"Error executing tool '{tool_name}': {str(e)}"
    
    def integrate_tool_result(
        self,
        query: str,
        tool_name: str,
        tool_input: str,
        tool_result: str,
        context: List[str]
    ) -> str:
        """
        Integrate tool results with RAG context to generate a final answer
        
        Args:
            query: The user's query
            tool_name: Name of the tool that was executed
            tool_input: Input that was passed to the tool
            tool_result: Result from the tool execution
            context: RAG context passages
            
        Returns:
            Integrated answer
        """
        # Create the prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an assistant that integrates tool results with RAG context.
            Generate a comprehensive answer that combines information from both the tool result
            and the RAG context.
            
            Make sure to:
            1. Clearly indicate when you're using information from the tool
            2. Provide a coherent answer that addresses the user's query
            3. Cite sources from the RAG context when appropriate
            """),
            ("human", """Query: {query}
            
            Tool Used: {tool_name}
            Tool Input: {tool_input}
            Tool Result: {tool_result}
            
            RAG Context:
            {context}
            
            Generate an integrated answer:""")
        ])
        
        # Format the context
        formatted_context = "\n\n".join(context)
        
        # Generate the response
        response = self.llm.invoke(
            prompt.format_messages(
                query=query,
                tool_name=tool_name,
                tool_input=tool_input,
                tool_result=tool_result,
                context=formatted_context
            )
        )
        
        return response.content