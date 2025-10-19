#!/usr/bin/env python3
"""
Advanced LLM generation pipeline with GPU optimization, model quantization, and context management.
Specifically designed for Chromium development queries with sophisticated prompting strategies.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    BitsAndBytesConfig, GenerationConfig,
    pipeline, TextStreamer
)
from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, PromptTemplate

from rag_system.core.config import get_config
from rag_system.core.logger import setup_logger
from rag_system.retrieval.retriever import RetrievalResult


class ModelSize(Enum):
    """Supported model sizes for different use cases."""
    SMALL = "small"    # 7B models - fast, good for simple queries
    MEDIUM = "medium"  # 13B models - balanced performance
    LARGE = "large"    # 70B+ models - best quality, slower


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_length: int = 2048
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for transformers."""
        return {
            'max_length': self.max_length,
            'max_new_tokens': self.max_new_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'repetition_penalty': self.repetition_penalty,
            'do_sample': self.do_sample,
            'pad_token_id': self.pad_token_id,
            'eos_token_id': self.eos_token_id
        }


@dataclass
class GenerationResult:
    """Result from LLM generation."""
    response: str
    query: str
    retrieved_contexts: List[RetrievalResult]
    generation_time: float
    token_count: int
    model_name: str
    context_length: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'response': self.response,
            'query': self.query,
            'retrieved_contexts': [ctx.to_dict() for ctx in self.retrieved_contexts],
            'generation_time': self.generation_time,
            'token_count': self.token_count,
            'model_name': self.model_name,
            'context_length': self.context_length,
            'metadata': self.metadata,
            'timestamp': datetime.now().isoformat()
        }


class ChromiumPromptManager:
    """Manages specialized prompts for Chromium development queries."""
    
    def __init__(self):
        self.logger = setup_logger(f"{__name__}.ChromiumPromptManager")
        
        # System prompts for different query types
        self.system_prompts = {
            'general': """You are an expert Chromium developer assistant with deep knowledge of the Chromium codebase, architecture, and development practices. You provide accurate, helpful answers about Chromium development based on the provided context from commit data.

Key guidelines:
- Focus on technical accuracy and practical implementation details
- Reference specific files, functions, and code patterns when relevant
- Explain the reasoning behind code changes and architectural decisions
- Consider performance, security, and maintainability implications
- Provide actionable guidance for developers

Always base your answers on the provided context and clearly indicate when information might be incomplete.""",

            'bug_fix': """You are a Chromium bug analysis expert. You help developers understand, diagnose, and fix bugs in the Chromium codebase.

When analyzing bugs:
- Identify the root cause and affected components
- Explain the technical details of what went wrong
- Suggest specific fixes with code examples when possible
- Consider edge cases and potential regressions
- Reference similar historical fixes from the context
- Discuss testing strategies to prevent similar issues

Focus on practical, actionable debugging and fixing guidance.""",

            'performance': """You are a Chromium performance optimization specialist. You help developers understand and improve performance across all aspects of Chromium.

When discussing performance:
- Identify performance bottlenecks and their impact
- Explain optimization techniques and their trade-offs
- Reference performance metrics and measurement approaches
- Consider memory usage, CPU efficiency, and GPU utilization
- Discuss caching strategies and resource management
- Explain the performance implications of architectural changes

Provide specific, measurable optimization recommendations.""",

            'security': """You are a Chromium security expert. You help developers understand security implications and implement secure coding practices.

When addressing security topics:
- Identify potential security vulnerabilities and attack vectors
- Explain security mitigations and their effectiveness
- Reference security best practices and coding patterns
- Discuss the principle of least privilege and sandboxing
- Consider both browser security and user privacy
- Explain security testing and validation approaches

Focus on proactive security measures and threat mitigation.""",

            'architecture': """You are a Chromium architecture specialist. You help developers understand system design, component interactions, and architectural decisions.

When discussing architecture:
- Explain component relationships and data flow
- Discuss design patterns and architectural principles
- Consider scalability, maintainability, and extensibility
- Reference architectural documentation and design docs
- Explain abstraction layers and interfaces
- Discuss the evolution of architectural decisions

Provide clear explanations of complex system interactions."""
        }
        
        # Context formatting templates
        self.context_template = """
## Relevant Chromium Context

{contexts}

---
"""
        
        self.context_item_template = """
### Context {index}: {title}
**File:** {file_path}
**Author:** {author} ({date})
**Type:** {content_type}
**Relevance:** {score:.3f}

```
{content}
```

"""
    
    def get_system_prompt(self, query_type: str = 'general') -> str:
        """Get system prompt for query type."""
        return self.system_prompts.get(query_type, self.system_prompts['general'])
    
    def format_context(self, retrieval_results: List[RetrievalResult]) -> str:
        """Format retrieval results into context for the prompt."""
        if not retrieval_results:
            return "## No specific context available\n\n"
        
        formatted_contexts = []
        
        for i, result in enumerate(retrieval_results, 1):
            document = result.search_result.document
            metadata = document.metadata or {}
            
            # Create a descriptive title
            file_path = metadata.get('file_path', 'Unknown file')
            commit_hash = metadata.get('commit_hash', 'Unknown commit')[:8]
            title = f"{file_path} ({commit_hash})"
            
            # Format the context item
            context_item = self.context_item_template.format(
                index=i,
                title=title,
                file_path=file_path,
                author=metadata.get('author', 'Unknown'),
                date=metadata.get('commit_date', 'Unknown date'),
                content_type=metadata.get('chunk_type', 'code'),
                score=result.search_result.score,
                content=document.content[:1000] + ('...' if len(document.content) > 1000 else '')
            )
            
            formatted_contexts.append(context_item)
        
        return self.context_template.format(contexts='\n'.join(formatted_contexts))
    
    def create_prompt(
        self, 
        query: str, 
        retrieval_results: List[RetrievalResult],
        query_type: str = 'general'
    ) -> str:
        """Create complete prompt with system message, context, and query."""
        system_prompt = self.get_system_prompt(query_type)
        context = self.format_context(retrieval_results)
        
        full_prompt = f"""<|system|>
{system_prompt}
<|end|>

{context}

<|user|>
{query}
<|end|>

<|assistant|>
"""
        
        return full_prompt


class LLMModelManager:
    """Manages LLM models with GPU optimization and quantization."""
    
    def __init__(self):
        self.config = get_config()
        self.logger = setup_logger(f"{__name__}.LLMModelManager")
        
        # Model configurations for different sizes
        self.model_configs = {
            ModelSize.SMALL: {
                'model_name': 'microsoft/DialoGPT-medium',  # Fallback for testing
                'quantization': '4bit',
                'max_memory_gb': 4
            },
            ModelSize.MEDIUM: {
                'model_name': 'microsoft/DialoGPT-large',   # Fallback for testing
                'quantization': '4bit', 
                'max_memory_gb': 8
            },
            ModelSize.LARGE: {
                'model_name': 'microsoft/DialoGPT-large',   # Fallback for testing
                'quantization': '8bit',
                'max_memory_gb': 16
            }
        }
        
        # Currently loaded model
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        self.device = self._setup_device()
        
        self.logger.info(f"Initialized LLM model manager on device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Set up the compute device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.logger.info(f"GPU available: {torch.cuda.get_device_name()}")
            self.logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            device = torch.device("cpu")
            self.logger.warning("CUDA not available, using CPU")
        
        return device
    
    def _create_quantization_config(self, quantization: str) -> Optional[BitsAndBytesConfig]:
        """Create quantization configuration."""
        if not torch.cuda.is_available():
            return None
        
        if quantization == '4bit':
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        elif quantization == '8bit':
            return BitsAndBytesConfig(
                load_in_8bit=True
            )
        
        return None
    
    def load_model(
        self, 
        model_size: ModelSize = ModelSize.SMALL,
        custom_model_name: Optional[str] = None
    ) -> bool:
        """Load and configure LLM model."""
        model_config = self.model_configs[model_size]
        model_name = custom_model_name or model_config['model_name']
        
        # Check if model is already loaded
        if self.current_model_name == model_name:
            self.logger.info(f"Model {model_name} already loaded")
            return True
        
        self.logger.info(f"Loading model: {model_name}")
        start_time = time.time()
        
        try:
            # Create quantization config
            quantization_config = self._create_quantization_config(
                model_config['quantization']
            )
            
            # Load tokenizer
            self.logger.info("Loading tokenizer...")
            self.current_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                padding_side='left'
            )
            
            # Set pad token if not present
            if self.current_tokenizer.pad_token is None:
                self.current_tokenizer.pad_token = self.current_tokenizer.eos_token
            
            # Load model with optimizations
            self.logger.info("Loading model with optimizations...")
            load_kwargs = {
                'pretrained_model_name_or_path': model_name,
                'trust_remote_code': True,
                'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32,
                'device_map': 'auto' if torch.cuda.is_available() else None
            }
            
            if quantization_config:
                load_kwargs['quantization_config'] = quantization_config
                self.logger.info(f"Using {model_config['quantization']} quantization")
            
            self.current_model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
            
            # Enable gradient checkpointing for memory efficiency
            if hasattr(self.current_model, 'gradient_checkpointing_enable'):
                self.current_model.gradient_checkpointing_enable()
            
            # Compile model for optimization (PyTorch 2.0+)
            if torch.__version__ >= '2.0' and torch.cuda.is_available():
                try:
                    self.current_model = torch.compile(self.current_model)
                    self.logger.info("Model compiled for optimization")
                except Exception as e:
                    self.logger.warning(f"Could not compile model: {e}")
            
            self.current_model_name = model_name
            elapsed = time.time() - start_time
            
            self.logger.info(f"Successfully loaded model in {elapsed:.2f}s")
            
            # Log memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1e9
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9
                self.logger.info(f"GPU memory usage: {memory_used:.1f}GB / {memory_total:.1f}GB")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    def generate(
        self,
        prompt: str,
        generation_config: GenerationConfig,
        stream: bool = False
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate text using the loaded model."""
        if not self.current_model or not self.current_tokenizer:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        start_time = time.time()
        
        # Tokenize input
        inputs = self.current_tokenizer.encode(
            prompt, 
            return_tensors='pt',
            truncation=True,
            max_length=4096  # Limit input length
        )
        
        if torch.cuda.is_available():
            inputs = inputs.to(self.device)
        
        # Set generation parameters
        generation_kwargs = generation_config.to_dict()
        generation_kwargs['pad_token_id'] = self.current_tokenizer.pad_token_id
        generation_kwargs['eos_token_id'] = self.current_tokenizer.eos_token_id
        
        # Generate
        with torch.inference_mode():
            if stream:
                # Streaming generation
                streamer = TextStreamer(
                    self.current_tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True
                )
                generation_kwargs['streamer'] = streamer
            
            outputs = self.current_model.generate(
                inputs,
                **generation_kwargs
            )
        
        # Decode output
        input_length = inputs.shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.current_tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )
        
        generation_time = time.time() - start_time
        
        # Generation metadata
        metadata = {
            'input_tokens': input_length,
            'output_tokens': len(generated_tokens),
            'total_tokens': input_length + len(generated_tokens),
            'generation_time': generation_time,
            'tokens_per_second': len(generated_tokens) / generation_time,
            'model_name': self.current_model_name
        }
        
        return response, metadata
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        if not self.current_model:
            return {'status': 'No model loaded'}
        
        info = {
            'model_name': self.current_model_name,
            'device': str(self.device),
            'model_dtype': str(self.current_model.dtype) if hasattr(self.current_model, 'dtype') else 'unknown'
        }
        
        if torch.cuda.is_available():
            info['gpu_memory_allocated'] = f"{torch.cuda.memory_allocated() / 1e9:.1f}GB"
            info['gpu_memory_reserved'] = f"{torch.cuda.memory_reserved() / 1e9:.1f}GB"
        
        return info


class AdvancedGenerator:
    """
    Advanced generation pipeline combining retrieval results with LLM generation.
    """
    
    def __init__(self, model_size: ModelSize = ModelSize.SMALL):
        """
        Initialize the advanced generator.
        
        Args:
            model_size: Size of the model to use
        """
        self.config = get_config()
        self.logger = setup_logger(f"{__name__}.AdvancedGenerator")
        
        # Initialize components
        self.model_manager = LLMModelManager()
        self.prompt_manager = ChromiumPromptManager()
        self.model_size = model_size
        
        # Generation configurations for different scenarios
        self.generation_configs = {
            'precise': GenerationConfig(
                max_new_tokens=256,
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.1
            ),
            'creative': GenerationConfig(
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                repetition_penalty=1.05
            ),
            'detailed': GenerationConfig(
                max_new_tokens=1024,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1
            )
        }
        
        self.logger.info("Initialized advanced generator")
    
    def initialize(self, custom_model_name: Optional[str] = None) -> bool:
        """Initialize the generator by loading the model."""
        self.logger.info("Initializing generator...")
        return self.model_manager.load_model(self.model_size, custom_model_name)
    
    def generate_response(
        self,
        query: str,
        retrieval_results: List[RetrievalResult],
        query_type: str = 'general',
        generation_style: str = 'precise',
        stream: bool = False
    ) -> GenerationResult:
        """
        Generate response based on query and retrieval results.
        
        Args:
            query: User query
            retrieval_results: Results from retrieval system
            query_type: Type of query (general, bug_fix, performance, etc.)
            generation_style: Generation style (precise, creative, detailed)
            stream: Whether to stream the response
            
        Returns:
            Generation result with response and metadata
        """
        self.logger.info(f"Generating response for {query_type} query: '{query[:50]}...'")
        
        start_time = time.time()
        
        # Create prompt
        prompt = self.prompt_manager.create_prompt(
            query, retrieval_results, query_type
        )
        
        # Get generation config
        gen_config = self.generation_configs.get(
            generation_style, 
            self.generation_configs['precise']
        )
        
        # Generate response
        response, gen_metadata = self.model_manager.generate(
            prompt, gen_config, stream
        )
        
        total_time = time.time() - start_time
        
        # Create result
        result = GenerationResult(
            response=response.strip(),
            query=query,
            retrieved_contexts=retrieval_results,
            generation_time=total_time,
            token_count=gen_metadata['total_tokens'],
            model_name=gen_metadata['model_name'],
            context_length=len(prompt),
            metadata={
                'query_type': query_type,
                'generation_style': generation_style,
                'prompt_length': len(prompt),
                'num_contexts': len(retrieval_results),
                **gen_metadata
            }
        )
        
        self.logger.info(f"Generated response in {total_time:.2f}s "
                        f"({gen_metadata['tokens_per_second']:.1f} tokens/sec)")
        
        return result
    
    def batch_generate(
        self,
        queries: List[Tuple[str, List[RetrievalResult]]],
        query_type: str = 'general',
        generation_style: str = 'precise'
    ) -> List[GenerationResult]:
        """Generate responses for multiple queries."""
        self.logger.info(f"Batch generating {len(queries)} responses")
        
        results = []
        for query, retrieval_results in queries:
            try:
                result = self.generate_response(
                    query, retrieval_results, query_type, generation_style
                )
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to generate response for '{query}': {e}")
                # Create error result
                error_result = GenerationResult(
                    response=f"Error generating response: {str(e)}",
                    query=query,
                    retrieved_contexts=retrieval_results,
                    generation_time=0.0,
                    token_count=0,
                    model_name="error",
                    context_length=0,
                    metadata={'error': str(e)}
                )
                results.append(error_result)
        
        return results
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation system statistics."""
        model_info = self.model_manager.get_model_info()
        
        stats = {
            'model_info': model_info,
            'model_size': self.model_size.value,
            'available_query_types': list(self.prompt_manager.system_prompts.keys()),
            'available_generation_styles': list(self.generation_configs.keys()),
            'prompt_manager': {
                'num_system_prompts': len(self.prompt_manager.system_prompts),
                'context_template_length': len(self.prompt_manager.context_template)
            }
        }
        
        return stats