import gzip
import json
from typing import List, Any
from memgpt.config import MemGPTConfig
from memgpt.data_types import LLMConfig, EmbeddingConfig
from memgpt.constants import LLM_MAX_TOKENS
import uuid
from datetime import datetime
import os


def load_gzipped_file(file_path):
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def read_jsonl(filename) -> List[dict]:
    lines = []
    with open(filename, "r") as file:
        for line in file:
            lines.append(json.loads(line.strip()))
    return lines


def get_experiment_config(postgres_uri, endpoint_type="openai", model="gpt-4"):
    from dotenv import load_dotenv
    load_dotenv()
    config = MemGPTConfig.load()
    config.archival_storage_type = "postgres"
    config.archival_storage_uri = postgres_uri

    if endpoint_type == "openai":
        llm_config = LLMConfig(
            model=model, 
            model_endpoint_type="openai", 
            # NOTE: this will override the model_endpoint in the config file
            model_endpoint=os.getenv("OPENAI_BASE_URL").strip(),
            context_window=8192
        )
        embedding_config = EmbeddingConfig(
            embedding_endpoint_type="hugging-face",
            embedding_endpoint=os.getenv("EMBEDDING_BASE_URL").strip(),
            embedding_dim=int(os.getenv("EMBEDDING_DIM").strip()),
            embedding_model=os.getenv("EMBEDDING_MODEL").strip(),
            embedding_chunk_size=300,  # TODO: fix this
        )
    else:
        assert model == "ehartford/dolphin-2.5-mixtral-8x7b", "Only model supported is ehartford/dolphin-2.5-mixtral-8x7b"
        llm_config = LLMConfig(
            model="ehartford/dolphin-2.5-mixtral-8x7b",
            model_endpoint_type="vllm",
            model_endpoint="https://api.memgpt.ai",
            model_wrapper="chatml",
            context_window=16384,
        )
        embedding_config = EmbeddingConfig(
            embedding_endpoint_type="hugging-face",
            embedding_endpoint="https://embeddings.memgpt.ai",
            embedding_dim=1024,
            embedding_model="BAAI/bge-large-en-v1.5",
            embedding_chunk_size=300,
        )

    config = MemGPTConfig(
        anon_clientid=config.anon_clientid,
        archival_storage_type="postgres",
        archival_storage_uri=postgres_uri,
        recall_storage_type="postgres",
        recall_storage_uri=postgres_uri,
        metadata_storage_type="postgres",
        metadata_storage_uri=postgres_uri,
        default_llm_config=llm_config,
        default_embedding_config=embedding_config,
    )
    print("Config model", config.default_llm_config.model)
    return config

def make_json_serializable(obj: Any) -> Any:
    """Recursively convert non-serializable objects to serializable format
    
    Handles:
    - UUID objects
    - datetime objects
    - LLMConfig objects (including by class name check)
    - EmbeddingConfig objects (including by class name check)
    - Pydantic models (with model_dump or dict methods)
    - Objects with __dict__ attributes
    - Namedtuples
    - Sets (converts to lists)
    """
    # Handle basic Python types first
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    if isinstance(obj, set):
        return [make_json_serializable(item) for item in obj]
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    
    # Check by class name as well (in case of import issues or subclasses)
    obj_class_name = type(obj).__name__
    
    # Handle LLMConfig objects (check by isinstance and class name)
    if isinstance(obj, LLMConfig) or obj_class_name == 'LLMConfig':
        try:
            return {
                "model": obj.model,
                "model_endpoint_type": obj.model_endpoint_type,
                "model_endpoint": obj.model_endpoint,
                "model_wrapper": obj.model_wrapper,
                "context_window": obj.context_window,
            }
        except AttributeError:
            # Fallback to __dict__ if attributes are not accessible
            return make_json_serializable(vars(obj))
    
    # Handle EmbeddingConfig objects (check by isinstance and class name)
    if isinstance(obj, EmbeddingConfig) or obj_class_name == 'EmbeddingConfig':
        try:
            return {
                "embedding_endpoint_type": obj.embedding_endpoint_type,
                "embedding_endpoint": obj.embedding_endpoint,
                "embedding_model": obj.embedding_model,
                "embedding_dim": obj.embedding_dim,
                "embedding_chunk_size": obj.embedding_chunk_size,
            }
        except AttributeError:
            # Fallback to __dict__ if attributes are not accessible
            return make_json_serializable(vars(obj))
    
    # Handle Pydantic models
    if hasattr(obj, 'model_dump'):
        try:
            return make_json_serializable(obj.model_dump())
        except (TypeError, AttributeError):
            pass
    
    if hasattr(obj, 'dict'):
        try:
            return make_json_serializable(obj.dict())
        except (TypeError, AttributeError):
            pass
    
    # Handle objects with __dict__ (like custom classes)
    if hasattr(obj, '__dict__'):
        try:
            if hasattr(obj, 'to_dict'):
                return make_json_serializable(obj.to_dict())
            # Try vars() first, which works with most objects
            return make_json_serializable(vars(obj))
        except (TypeError, AttributeError):
            pass
    
    # Handle namedtuples
    if hasattr(obj, '_asdict'):
        try:
            return make_json_serializable(obj._asdict())
        except (TypeError, AttributeError):
            pass
    
    # Last resort: try to serialize normally
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        # If it can't be serialized, convert to string representation
        return str(obj)