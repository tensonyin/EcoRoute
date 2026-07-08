import os
import sys
import json
import time
import re
import requests

def check_env():
    """
    1. Environment Interceptor
    Assert presence of FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS.
    Exit with code 10 on failure.
    """
    required = ["FIREWORKS_API_KEY", "FIREWORKS_BASE_URL", "ALLOWED_MODELS"]
    missing = []
    for var in required:
        if var not in os.environ or not os.environ[var].strip():
            missing.append(var)
            
    if missing:
        sys.stderr.write(f"CRITICAL ERROR: Missing required environment variables: {', '.join(missing)}\n")
        sys.stderr.flush()
        sys.exit(10)
    
    api_key = os.environ["FIREWORKS_API_KEY"].strip()
    base_url = os.environ["FIREWORKS_BASE_URL"].strip()
    allowed_models = [m.strip() for m in os.environ["ALLOWED_MODELS"].split(",") if m.strip()]
    
    if not allowed_models:
        sys.stderr.write("CRITICAL ERROR: ALLOWED_MODELS variable is empty.\n")
        sys.stderr.flush()
        sys.exit(10)
        
    return api_key, base_url, allowed_models

def map_allowed_models(allowed_models):
    """
    Maps allowed models to routing categories.
    If target models are missing, fallback intelligently.
    """
    mapping = {
        "cheap": None,      # gemma-4-26b-a4b-it
        "mid_dense": None,  # gemma-4-31b-it
        "mid_quant": None,  # gemma-4-31b-it-nvfp4
        "code": None,       # kimi-k2p7-code
        "flagship": None    # minimax-m3
    }
    
    for m in allowed_models:
        m_lower = m.lower()
        if "26b" in m_lower or "cheap" in m_lower or "small" in m_lower or "mini" in m_lower:
            mapping["cheap"] = m
        elif "nvfp4" in m_lower:
            mapping["mid_quant"] = m
        elif "31b" in m_lower or "glm" in m_lower:
            mapping["mid_dense"] = m
        elif "kimi" in m_lower or "code" in m_lower:
            mapping["code"] = m
        elif "minimax" in m_lower or "m3" in m_lower or "deepseek" in m_lower or "gpt" in m_lower:
            mapping["flagship"] = m
            
    default_model = allowed_models[0]
    
    # Resolve fallbacks
    if not mapping["cheap"]:
        mapping["cheap"] = next((m for m in allowed_models if "gemma" in m.lower()), default_model)
        
    if not mapping["mid_quant"]:
        mapping["mid_quant"] = mapping["mid_dense"] or next((m for m in allowed_models if "gemma" in m.lower()), default_model)
        
    if not mapping["mid_dense"]:
        mapping["mid_dense"] = mapping["mid_quant"] or next((m for m in allowed_models if "gemma" in m.lower()), default_model)
        
    if not mapping["code"]:
        mapping["code"] = next((m for m in allowed_models if "kimi" in m.lower() or "code" in m.lower()), default_model)
        
    if not mapping["flagship"]:
        mapping["flagship"] = next((m for m in allowed_models if "minimax" in m.lower() or "m3" in m.lower() or "deepseek" in m_lower or "gpt" in m_lower), default_model)
        
    return mapping

def clean_prompt(prompt):
    """
    Input compression (Stage 3): Collapses extra whitespace, strips politeness,
    and simplifies redundant instruction prefixes to minimize input token usage.
    """
    if not isinstance(prompt, str):
        return prompt
        
    # Step 1: Strip politeness and requests filler
    polite_pats = [
        (r'\b(?:please|kindly|could you|would you|can you|help me(?:\s+to)?|i want you to)\b,?\s*', ''),
    ]
    for pat, repl in polite_pats:
        prompt = re.sub(pat, repl, prompt, flags=re.IGNORECASE)
        
    # Step 2: Simplify common instruction prefixes to direct short commands
    instruction_pats = [
        # Sentiment
        (r'\bclassify (?:the )?sentiment of (?:the following |this |the )?(?:sentence|text|review)?\b\s*:?\s*', 'classify sentiment: '),
        # Summarisation
        (r'\bsummarize (?:the following |this |the )?(?:text|article|essay|passage)?\b\s*:?\s*', 'summarize: '),
        # NER
        (r'\bextract (?:the |all )?(?:named )?entities (?:from|in) (?:the following |this |the )?(?:text|sentence|passage)?\b\s*:?\s*', 'extract entities: '),
        # Code Debugging
        (r'\b(?:correct|fix) (?:the |all )?(?:bug|error|syntax error|runtime error|exception|issue)?s?\s*(?:in|of)\s*(?:the following |this |the )?(?:code|function|implementation)?\b\s*:?\s*', 'fix: '),
        # Code Generation
        (r'\b(?:write|create|implement|generate) a (?:python|c\+\+|javascript|java|rust|go|html|css)?\s*(?:function|script|code|program|class)\b\s*', 'code '),
    ]
    for pat, repl in instruction_pats:
        prompt = re.sub(pat, repl, prompt, flags=re.IGNORECASE)
        
    # Step 3: Collapse whitespace and limit newlines
    prompt = prompt.strip()
    prompt = re.sub(r'[ \t]+', ' ', prompt)
    prompt = re.sub(r'\n{3,}', '\n\n', prompt)
    
    return prompt

def classify_task(prompt):
    """
    Heuristic task classification and routing.
    Returns: category, system_prompt, target_model_key, max_tokens
    """
    prompt_lower = prompt.lower()
    
    # 3. Sentiment Classification
    if re.search(r'\b(sentiment|positive|negative|neutral|classify the tone|review tone|emotional tone|emotion)\b', prompt_lower) or "sentiment" in prompt_lower:
        return "sentiment", "Sentiment: label + brief justification.", "cheap", 100
        
    # 4. Text Summarisation
    if re.search(r'\b(summarize|summary|summarise|condensation|condense|abstract|tl;dr|tldr)\b', prompt_lower):
        return "summarisation", "Summarize text concisely.", "cheap", 150
        
    # 5. Named Entity Recognition (NER)
    if re.search(r'\b(extract|identify|find)\b.*\b(entities|entity|names|person|org|location|date|places|organizations)\b', prompt_lower) or "ner" in prompt_lower or "entity recognition" in prompt_lower:
        return "ner", "Extract entities (Person,Org,Loc,Date).", "cheap", 120
        
    # 6. Code Debugging
    if re.search(r'\b(bug|debug|fix|correct|troubleshoot|syntax error|runtime error|exception)\b.*\b(code|python|function|c\+\+|javascript|java|implementation)\b', prompt_lower) or "correct the implementation" in prompt_lower:
        return "code_debug", "Output only corrected code.", "code", 512
        
    # 8. Code Generation
    if re.search(r'\b(write|create|implement|generate|code|program|script|function|class)\b.*\b(code|python|c\+\+|javascript|java|rust|go|html|css)\b', prompt_lower) or "write a function" in prompt_lower:
        return "code_generation", "Output only code.", "code", 512
        
    # 7. Logic Puzzles
    if re.search(r'\b(logic|puzzle|deduce|deduction|reasoning|riddle|constraint|grid puzzle|truth-teller|liar)\b', prompt_lower) or "if " in prompt_lower:
        return "logic_puzzles", "Solve logic puzzle concisely.", "mid_dense", 384
        
    # 2. Mathematical Reasoning
    if re.search(r'\b(solve|calculate|compute|equation|algebra|arithmetic|percentage|probability|ratio|sum|product|difference|fraction)\b', prompt_lower) or "math" in prompt_lower:
        return "math_reasoning", "Solve math concisely step-by-step.", "mid_dense", 256
        
    # 1. Factual Q&A / Fallback
    return "factual_qa", "Direct factual answer only.", "flagship", 300

def stream_tasks(file_path):
    """
    2. JSON Input Parser
    Low-memory streaming JSON array generator.
    Yields dicts with 'task_id' and 'prompt'.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        brace_count = 0
        in_string = False
        escape = False
        buffer = []
        
        while True:
            chunk = f.read(65536) # read 64KB chunks
            if not chunk:
                break
                
            for char in chunk:
                if brace_count > 0:
                    buffer.append(char)
                    
                if escape:
                    escape = False
                    continue
                    
                if char == '\\':
                    escape = True
                    continue
                    
                if char == '"':
                    in_string = not in_string
                    continue
                    
                if not in_string:
                    if char == '{':
                        if brace_count == 0:
                            buffer = ['{']
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            obj_str = "".join(buffer)
                            try:
                                task = json.loads(obj_str)
                                if "task_id" in task and "prompt" in task:
                                    yield task
                                else:
                                    sys.stderr.write(f"WARNING: Skipping invalid task format: {obj_str[:100]}...\n")
                            except Exception as parse_err:
                                sys.stderr.write(f"WARNING: Failed to parse task JSON: {parse_err}\n")
                            buffer = []

def request_with_retry(prompt, sys_prompt, model, api_key, base_url, max_tokens, timeout=25, max_retries=3):
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens
    }
    
    backoff = 1.0
    for attempt in range(max_retries):
        try:
            sys.stderr.write(f"Sending request to {model} (Attempt {attempt+1}/{max_retries})...\n")
            sys.stderr.flush()
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            
            if response.status_code == 200:
                data = response.json()
                choice = data["choices"][0]
                message = choice.get("message", {})
                content = message.get("content")
                if content is None:
                    content = message.get("reasoning_content")
                if content is None:
                    content = ""
                return content
            elif response.status_code == 429:
                sys.stderr.write(f"Rate limited (429) on model {model}. Retrying in {backoff}s...\n")
            elif response.status_code >= 500:
                sys.stderr.write(f"Server error ({response.status_code}) on model {model}. Retrying in {backoff}s...\n")
            else:
                response.raise_for_status()
                
        except (requests.RequestException, KeyError, ValueError) as err:
            sys.stderr.write(f"Request failed: {err}. Retrying in {backoff}s...\n")
            
        sys.stderr.flush()
        time.sleep(backoff)
        backoff *= 2.0
        
    raise RuntimeError(f"All retries failed for model {model}")

def process_task(task, api_key, base_url, model_mapping, allowed_models):
    task_id = task["task_id"]
    prompt = clean_prompt(task["prompt"])
    
    category, sys_prompt, target_key, max_tokens = classify_task(prompt)
    
    primary_model = model_mapping[target_key]
    backup_model = model_mapping["flagship"] or allowed_models[0]
    
    if primary_model == backup_model and len(allowed_models) > 1:
        backup_model = next((m for m in allowed_models if m != primary_model), allowed_models[0])
        
    try:
        answer = request_with_retry(prompt, sys_prompt, primary_model, api_key, base_url, max_tokens)
        return {"task_id": task_id, "answer": answer}
    except Exception as e:
        sys.stderr.write(f"Primary model {primary_model} failed for task {task_id} ({category}): {e}. Trying backup model...\n")
        sys.stderr.flush()
        try:
            answer = request_with_retry(prompt, sys_prompt, backup_model, api_key, base_url, max_tokens)
            return {"task_id": task_id, "answer": answer}
        except Exception as e2:
            sys.stderr.write(f"Backup model {backup_model} failed for task {task_id}: {e2}. Using fallback answer.\n")
            sys.stderr.flush()
            return {"task_id": task_id, "answer": "Failed to generate answer due to upstream error."}

def write_results_atomic(results, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    tmp_path = f"{output_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
        
    os.replace(tmp_path, output_path)

def main():
    start_time = time.time()
    
    # 1. Environment Interceptor
    api_key, base_url, allowed_models = check_env()
    
    # Build the routing model map
    model_mapping = map_allowed_models(allowed_models)
    
    input_path = os.environ.get("INPUT_TASKS_PATH", "/input/tasks.json")
    output_path = os.environ.get("OUTPUT_RESULTS_PATH", "/output/results.json")
    
    if not os.path.exists(input_path):
        sys.stderr.write(f"CRITICAL ERROR: Input file not found at {input_path}\n")
        sys.stderr.flush()
        sys.exit(1)
        
    results = []
    
    # 2. JSON Input Parser
    try:
        for task in stream_tasks(input_path):
            # Check elapsed time (9.5 minutes threshold)
            elapsed = time.time() - start_time
            if elapsed > 570:
                sys.stderr.write("WARNING: Reached 9.5 minute time limit. Aborting further tasks to save current results.\n")
                sys.stderr.flush()
                break
                
            # 3. Exception Shield with routing and processing
            result = process_task(task, api_key, base_url, model_mapping, allowed_models)
            results.append(result)
            
    except Exception as stream_err:
        sys.stderr.write(f"CRITICAL ERROR: Unexpected error during task processing loop: {stream_err}\n")
        sys.stderr.flush()
        
    # 4. Compliant JSON Writer
    try:
        write_results_atomic(results, output_path)
    except Exception as write_err:
        sys.stderr.write(f"CRITICAL ERROR: Failed to write results atomically: {write_err}\n")
        sys.stderr.flush()
        sys.exit(1)
        
    sys.stderr.write(f"Pipeline completed successfully. Total processed tasks: {len(results)}\n")
    sys.stderr.flush()
    sys.exit(0)

if __name__ == "__main__":
    main()
