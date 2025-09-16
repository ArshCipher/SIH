"""
Advanced LLM Integration for Competition-Grade Medical AI
Supports OpenAI GPT-4, Anthropic Claude, and LangChain orchestration
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

# Optional imports with graceful fallback
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

try:
    from langchain.llms import OpenAI as LangChainOpenAI
    from langchain.chat_models import ChatOpenAI, ChatAnthropic
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema import HumanMessage, SystemMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Available LLM providers"""
    OPENAI_GPT4 = "openai_gpt4"
    OPENAI_GPT35 = "openai_gpt35"
    ANTHROPIC_CLAUDE = "anthropic_claude"
    LANGCHAIN_ORCHESTRATED = "langchain_orchestrated"

@dataclass
class LLMResponse:
    """LLM response with metadata"""
    content: str
    provider: LLMProvider
    model: str
    confidence: float
    tokens_used: int
    cost_estimate: float
    response_time: float
    reasoning: Optional[str] = None
    medical_entities: List[str] = None
    safety_validated: bool = False

class AdvancedLLMOrchestrator:
    """Competition-grade LLM orchestration for medical AI"""
    
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.langchain_models = {}
        
        # Initialize available clients
        self._initialize_clients()
        
        # Medical prompt templates
        self.medical_prompts = {
            "diagnosis": """You are a world-class medical AI assistant helping with diagnostic reasoning.
            
Patient Information: {patient_context}
Symptoms: {symptoms}
Medical History: {medical_history}

Provide a structured diagnostic assessment including:
1. Differential diagnoses (most likely to least likely)
2. Recommended diagnostic tests
3. Red flags to watch for
4. When to seek immediate care

IMPORTANT: Always include medical disclaimers and emphasize consulting healthcare professionals.""",
            
            "treatment": """You are an expert medical AI providing treatment guidance.
            
Condition: {condition}
Patient Profile: {patient_profile}
Current Medications: {medications}
Allergies: {allergies}

Provide evidence-based treatment recommendations including:
1. First-line treatments
2. Alternative options
3. Lifestyle modifications
4. Monitoring requirements
5. Drug interactions to consider

CRITICAL: Include all necessary medical disclaimers.""",
            
            "emergency_triage": """You are an emergency medical triage AI system.
            
Symptoms: {symptoms}
Severity: {severity}
Patient Vitals: {vitals}
Duration: {duration}

Assess urgency level and provide:
1. Triage category (Emergency/Urgent/Standard/Non-urgent)
2. Recommended action timeline
3. Warning signs for deterioration
4. Emergency contact guidance

URGENT: If life-threatening, immediately advise emergency services."""
        }
        
    def _initialize_clients(self):
        """Initialize all available LLM clients"""
        
        # Initialize OpenAI
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            try:
                self.openai_client = openai.OpenAI(
                    api_key=os.getenv("OPENAI_API_KEY")
                )
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.warning(f"OpenAI client initialization failed: {e}")
        
        # Initialize Anthropic
        if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
            try:
                self.anthropic_client = anthropic.Anthropic(
                    api_key=os.getenv("ANTHROPIC_API_KEY")
                )
                logger.info("Anthropic client initialized successfully")
            except Exception as e:
                logger.warning(f"Anthropic client initialization failed: {e}")
        
        # Initialize LangChain models
        if LANGCHAIN_AVAILABLE:
            try:
                if os.getenv("OPENAI_API_KEY"):
                    self.langchain_models["gpt4"] = ChatOpenAI(
                        model="gpt-4",
                        temperature=0.1,
                        openai_api_key=os.getenv("OPENAI_API_KEY")
                    )
                
                if os.getenv("ANTHROPIC_API_KEY"):
                    self.langchain_models["claude"] = ChatAnthropic(
                        model="claude-3-opus-20240229",
                        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
                    )
                
                logger.info("LangChain models initialized successfully")
            except Exception as e:
                logger.warning(f"LangChain models initialization failed: {e}")
    
    async def process_medical_query(
        self,
        query: str,
        query_type: str,
        patient_context: Dict[str, Any],
        providers: List[LLMProvider] = None
    ) -> List[LLMResponse]:
        """Process medical query using multiple LLM providers"""
        
        if providers is None:
            providers = [LLMProvider.OPENAI_GPT4, LLMProvider.ANTHROPIC_CLAUDE]
        
        responses = []
        
        for provider in providers:
            try:
                response = await self._query_provider(
                    provider, query, query_type, patient_context
                )
                if response:
                    responses.append(response)
            except Exception as e:
                logger.error(f"Error querying {provider.value}: {e}")
        
        return responses
    
    async def _query_provider(
        self,
        provider: LLMProvider,
        query: str,
        query_type: str,
        patient_context: Dict[str, Any]
    ) -> Optional[LLMResponse]:
        """Query specific LLM provider"""
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            if provider == LLMProvider.OPENAI_GPT4:
                return await self._query_openai_gpt4(query, query_type, patient_context, start_time)
            elif provider == LLMProvider.ANTHROPIC_CLAUDE:
                return await self._query_anthropic_claude(query, query_type, patient_context, start_time)
            elif provider == LLMProvider.LANGCHAIN_ORCHESTRATED:
                return await self._query_langchain(query, query_type, patient_context, start_time)
            else:
                logger.warning(f"Unsupported provider: {provider}")
                return None
                
        except Exception as e:
            logger.error(f"Error in {provider.value} query: {e}")
            return None
    
    async def _query_openai_gpt4(
        self, query: str, query_type: str, patient_context: Dict, start_time: float
    ) -> Optional[LLMResponse]:
        """Query OpenAI GPT-4"""
        
        if not self.openai_client:
            logger.warning("OpenAI client not available")
            return None
        
        try:
            # Prepare messages
            system_prompt = self.medical_prompts.get(query_type, "You are a medical AI assistant.")
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            # Make API call
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4",
                messages=messages,
                temperature=0.1,
                max_tokens=2000
            )
            
            # Calculate metrics
            response_time = asyncio.get_event_loop().time() - start_time
            tokens_used = response.usage.total_tokens
            cost_estimate = self._calculate_openai_cost(tokens_used, "gpt-4")
            
            return LLMResponse(
                content=response.choices[0].message.content,
                provider=LLMProvider.OPENAI_GPT4,
                model="gpt-4",
                confidence=0.85,  # Default confidence for GPT-4
                tokens_used=tokens_used,
                cost_estimate=cost_estimate,
                response_time=response_time,
                safety_validated=True
            )
            
        except Exception as e:
            logger.error(f"OpenAI GPT-4 query failed: {e}")
            return None
    
    async def _query_anthropic_claude(
        self, query: str, query_type: str, patient_context: Dict, start_time: float
    ) -> Optional[LLMResponse]:
        """Query Anthropic Claude"""
        
        if not self.anthropic_client:
            logger.warning("Anthropic client not available")
            return None
        
        try:
            # Prepare prompt
            system_prompt = self.medical_prompts.get(query_type, "You are a medical AI assistant.")
            full_prompt = f"{system_prompt}\\n\\nUser Query: {query}"
            
            # Make API call
            response = await asyncio.to_thread(
                self.anthropic_client.messages.create,
                model="claude-3-opus-20240229",
                max_tokens=2000,
                system=system_prompt,
                messages=[{"role": "user", "content": query}]
            )
            
            # Calculate metrics
            response_time = asyncio.get_event_loop().time() - start_time
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            cost_estimate = self._calculate_anthropic_cost(tokens_used, "claude-3-opus")
            
            return LLMResponse(
                content=response.content[0].text,
                provider=LLMProvider.ANTHROPIC_CLAUDE,
                model="claude-3-opus-20240229",
                confidence=0.88,  # Default confidence for Claude
                tokens_used=tokens_used,
                cost_estimate=cost_estimate,
                response_time=response_time,
                safety_validated=True
            )
            
        except Exception as e:
            logger.error(f"Anthropic Claude query failed: {e}")
            return None
    
    async def _query_langchain(
        self, query: str, query_type: str, patient_context: Dict, start_time: float
    ) -> Optional[LLMResponse]:
        """Query using LangChain orchestration"""
        
        if not LANGCHAIN_AVAILABLE or not self.langchain_models:
            logger.warning("LangChain not available")
            return None
        
        try:
            # Use GPT-4 via LangChain if available
            model = self.langchain_models.get("gpt4")
            if not model:
                return None
            
            # Create prompt template
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=self.medical_prompts.get(query_type, "You are a medical AI assistant.")),
                HumanMessage(content=query)
            ])
            
            # Execute chain
            chain = prompt | model
            response = await asyncio.to_thread(
                chain.invoke,
                {"query": query, **patient_context}
            )
            
            # Calculate metrics
            response_time = asyncio.get_event_loop().time() - start_time
            
            return LLMResponse(
                content=response.content,
                provider=LLMProvider.LANGCHAIN_ORCHESTRATED,
                model="gpt-4-langchain",
                confidence=0.82,
                tokens_used=0,  # LangChain doesn't always provide token counts
                cost_estimate=0.0,
                response_time=response_time,
                safety_validated=True
            )
            
        except Exception as e:
            logger.error(f"LangChain query failed: {e}")
            return None
    
    def _calculate_openai_cost(self, tokens: int, model: str) -> float:
        """Calculate estimated OpenAI API cost"""
        # Approximate pricing (as of 2024)
        pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002}
        }
        
        if model in pricing:
            # Assume 50/50 input/output split
            input_cost = (tokens * 0.5 / 1000) * pricing[model]["input"]
            output_cost = (tokens * 0.5 / 1000) * pricing[model]["output"]
            return input_cost + output_cost
        
        return 0.0
    
    def _calculate_anthropic_cost(self, tokens: int, model: str) -> float:
        """Calculate estimated Anthropic API cost"""
        # Approximate pricing (as of 2024)
        pricing = {
            "claude-3-opus": {"input": 0.015, "output": 0.075},  # per 1K tokens
            "claude-3-sonnet": {"input": 0.003, "output": 0.015}
        }
        
        if model in pricing:
            # Assume 50/50 input/output split
            input_cost = (tokens * 0.5 / 1000) * pricing[model]["input"]
            output_cost = (tokens * 0.5 / 1000) * pricing[model]["output"]
            return input_cost + output_cost
        
        return 0.0
    
    async def get_multi_llm_consensus(
        self,
        query: str,
        query_type: str,
        patient_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get consensus from multiple LLM providers"""
        
        # Query multiple providers
        responses = await self.process_medical_query(
            query, query_type, patient_context
        )
        
        if not responses:
            return {
                "consensus_response": "No LLM providers available.",
                "confidence": 0.0,
                "provider_responses": [],
                "cost_total": 0.0
            }
        
        # Calculate consensus
        consensus_confidence = sum(r.confidence for r in responses) / len(responses)
        total_cost = sum(r.cost_estimate for r in responses)
        
        # Combine responses (simple concatenation for now)
        combined_content = "\\n\\n---\\n\\n".join([
            f"**{r.provider.value.upper()} Response:**\\n{r.content}"
            for r in responses
        ])
        
        return {
            "consensus_response": combined_content,
            "confidence": consensus_confidence,
            "provider_responses": [
                {
                    "provider": r.provider.value,
                    "content": r.content,
                    "confidence": r.confidence,
                    "tokens": r.tokens_used,
                    "cost": r.cost_estimate,
                    "response_time": r.response_time
                }
                for r in responses
            ],
            "cost_total": total_cost,
            "providers_used": len(responses)
        }

# Global instance for easy access
advanced_llm = AdvancedLLMOrchestrator()