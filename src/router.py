import re
import sys

def map_allowed_models(allowed_models):
    """
    Maps allowed models to routing categories dynamically using substring matching.
    Supports:
      - cheap (e.g. 26b, 8b, 9b, small, lite, a4b)
      - code (e.g. code, coder, kimi)
      - mid_dense (e.g. 31b, 70b, instruct, glm)
      - flagship (e.g. minimax, m3, deepseek, gpt, 120b)
      
    If specific features are missing, falls back to the first available model.
    If only one model is provided, all categories map to it.
    """
    mapping = {
        "cheap": None,
        "mid_dense": None,
        "code": None,
        "flagship": None
    }
    
    if not allowed_models:
        return mapping

    # Defense A: Single model case
    if len(allowed_models) == 1:
        single = allowed_models[0]
        return {k: single for k in mapping.keys()}
        
    for m in allowed_models:
        m_lower = m.lower()
        # 1. Code expert
        if any(kw in m_lower for kw in ["code", "coder", "kimi"]):
            if not mapping["code"]:
                mapping["code"] = m
        # 2. Cheap / Economic model
        elif any(kw in m_lower for kw in ["26b", "8b", "9b", "small", "lite", "a4b"]):
            if not mapping["cheap"]:
                mapping["cheap"] = m
        # 3. Medium reasoning / Dense
        elif any(kw in m_lower for kw in ["31b", "70b", "instruct", "glm"]):
            if not mapping["mid_dense"]:
                mapping["mid_dense"] = m
        # 4. Premium / Flagship
        elif any(kw in m_lower for kw in ["minimax", "m3", "deepseek", "gpt", "120b"]):
            if not mapping["flagship"]:
                mapping["flagship"] = m

    default_model = allowed_models[0]

    # Resolve fallbacks dynamically using available models in priority order
    if not mapping["cheap"]:
        # Find any gemma or any model with 'small'/'26b', fallback to default
        mapping["cheap"] = next((m for m in allowed_models if "gemma" in m.lower()), default_model)
        
    if not mapping["mid_dense"]:
        # Fallback to flagship, then cheap, then default
        mapping["mid_dense"] = mapping["flagship"] or mapping["cheap"] or default_model
        
    if not mapping["code"]:
        # Fallback to flagship or default
        mapping["code"] = mapping["flagship"] or default_model
        
    if not mapping["flagship"]:
        # Fallback to first available model
        mapping["flagship"] = default_model
        
    return mapping

def should_route_local(category):
    """
    Decides in real-time if a task should be processed locally
    to save Fireworks token cost (0 Fireworks tokens).
    Currently routes Sentiment, Summarisation, and NER locally.
    """
    # Simple, highly-accurate NLP tasks can be solved by local 1.5B model
    return category in ["sentiment", "summarisation", "ner"]
