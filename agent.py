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

from src.router import map_allowed_models

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
    Returns: category, system_prompt, target_model_key, max_tokens, stop_sequences
    """
    prompt_lower = prompt.lower()
    
    # Standard strict constraints to eliminate CoT / Markdown formatting overhead
    strict_suffix = (
        " CRITICAL: Respond in PLAIN TEXT only. Do NOT think out loud. "
        "Do NOT use Markdown tables, bullet points, headers, or bold text. "
        "Do NOT include any introductory reasoning, explanations, or intermediate steps. "
        "Output the final answer directly."
    )
    
    # Stop sequences to block preamble / CoT leaking
    common_stops = ["We need to", "The text:", "Something like", "Entities:", "\n\n"]
    
    # 3. Sentiment Classification
    if re.search(r'\b(sentiment|positive|negative|neutral|classify the tone|review tone|emotional tone|emotion)\b', prompt_lower) or "sentiment" in prompt_lower:
        return "sentiment", f"Sentiment: label + brief justification.{strict_suffix}", "cheap", 15, common_stops
        
    # 4. Text Summarisation
    if re.search(r'\b(summarize|summary|summarise|condensation|condense|abstract|tl;dr|tldr)\b', prompt_lower):
        return "summarisation", f"Summarize text concisely.{strict_suffix}", "cheap", 100, common_stops
        
    # 5. Named Entity Recognition (NER)
    if re.search(r'\b(extract|identify|find)\b.*\b(entities|entity|names|person|org|location|date|places|organizations)\b', prompt_lower) or "ner" in prompt_lower or "entity recognition" in prompt_lower:
        return "ner", f"Extract entities (Person,Org,Loc,Date).{strict_suffix}", "cheap", 100, common_stops
        
    # 6. Code Debugging
    if re.search(r'\b(bug|debug|fix|correct|troubleshoot|syntax error|runtime error|exception)\b.*\b(code|python|function|c\+\+|javascript|java|implementation)\b', prompt_lower) or "correct the implementation" in prompt_lower:
        sys_code_suffix = (
            " CRITICAL: Output ONLY the raw executable code. "
            "Do NOT include markdown backticks (like ```python) or any explanations."
        )
        return "code_debug", f"Output only corrected code.{sys_code_suffix}", "code", 512, None
        
    # 8. Code Generation
    if re.search(r'\b(write|create|implement|generate|code|program|script|function|class)\b.*\b(code|python|c\+\+|javascript|java|rust|go|html|css)\b', prompt_lower) or "write a function" in prompt_lower:
        sys_code_suffix = (
            " CRITICAL: Output ONLY the raw executable code. "
            "Do NOT include markdown backticks (like ```python) or any explanations."
        )
        return "code_generation", f"Output only code.{sys_code_suffix}", "code", 512, None
        
    # 7. Logic Puzzles
    if re.search(r'\b(logic|puzzle|deduce|deduction|reasoning|riddle|constraint|grid puzzle|truth-teller|liar)\b', prompt_lower) or "if " in prompt_lower:
        return "logic_puzzles", f"Solve logic puzzle concisely.{strict_suffix}", "mid_dense", 256, None
        
    # 2. Mathematical Reasoning
    if re.search(r'\b(solve|calculate|compute|equation|algebra|arithmetic|percentage|probability|ratio|sum|product|difference|fraction)\b', prompt_lower) or "math" in prompt_lower:
        return "math_reasoning", f"Solve math concisely. Output only the final key results.{strict_suffix}", "mid_dense", 100, None
        
    # 1. Factual Q&A / Fallback
    return "factual_qa", f"Direct factual answer only.{strict_suffix}", "flagship", 200, None

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
llm = None

def get_local_model():
    global llm
    if llm is not None:
        return llm
        
    model_path = os.environ.get("LOCAL_MODEL_PATH", "model/gemma-4-E2B-it-Q4_K_M.gguf")
    if not os.path.exists(model_path):
        sys.stderr.write(f"Local model not found at {model_path}. Fallback to Fireworks.\n")
        sys.stderr.flush()
        return None
        
    try:
        from llama_cpp import Llama
        sys.stderr.write(f"Loading local model from {model_path}...\n")
        sys.stderr.flush()
        llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=2,
            verbose=False
        )
        sys.stderr.write("Local model loaded successfully!\n")
        sys.stderr.flush()
        return llm
    except Exception as e:
        sys.stderr.write(f"Failed to load local model: {e}\n")
        sys.stderr.flush()
        return None

def query_local_model(prompt, sys_prompt, max_tokens, stop_sequences=None):
    local_llm = get_local_model()
    if local_llm is None:
        raise RuntimeError("Local model is not available")
        
    # Check if this is a gemma model by path
    model_path = os.environ.get("LOCAL_MODEL_PATH", "model/gemma-4-E2B-it-Q4_K_M.gguf")
    is_gemma4 = "gemma" in model_path.lower()
    
    if is_gemma4:
        # Prompt for Gemma 4 using system and user roles, and ending in the native thought turn
        formatted_prompt = (
            f"<|turn|>system\n{sys_prompt}<turn|>\n"
            f"<|turn|>user\n{prompt}<turn|>\n"
            f"<|channel>thought\n"
        )
        # We do NOT include `<channel|>` or `<|turn|>model` in the stop sequences,
        # so the model can finish generating thoughts and transition to the model answer turn.
        # We stop at `<turn|>` (end of model turn) or `<|turn|>` (beginning of next turn).
        stop = ["<turn|>", "<|turn|>"]
    else:
        # Fallback to standard ChatML format (e.g. Qwen)
        formatted_prompt = (
            f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        stop = ["<|im_end|>", "<|im_start|>"]
        
    if stop_sequences:
        for s in stop_sequences:
            if s not in stop:
                stop.append(s)
                
    response = local_llm(
        formatted_prompt,
        max_tokens=max_tokens,
        stop=stop,
        temperature=0.0
    )
    
    answer = response["choices"][0]["text"].strip()
    
    # Dynamic CoT Isolation: If thoughts were generated, extract only the clean final answer
    if is_gemma4:
        if "<|turn|>model\n" in answer:
            answer = answer.split("<|turn|>model\n")[-1].strip()
        elif "model\n" in answer:
            answer = answer.split("model\n")[-1].strip()
            
        for tag in ["<turn|>", "<|turn|>", "<channel|>", "<|channel>thought"]:
            answer = answer.replace(tag, "").strip()
            
    return answer

def request_with_retry(prompt, sys_prompt, model, api_key, base_url, max_tokens, stop_sequences=None, timeout=25, max_retries=3):
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    backoff = 1.0
    attempt = 0
    while attempt < max_retries:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens
        }
        if stop_sequences:
            payload["stop"] = stop_sequences
            
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
        attempt += 1
        
    raise RuntimeError(f"All retries failed for model {model}")

def process_task(task, api_key, base_url, model_mapping, allowed_models):
    task_id = task["task_id"]
    prompt = clean_prompt(task["prompt"])
    
    category, sys_prompt, target_key, max_tokens, stop_sequences = classify_task(prompt)
    
    primary_model = model_mapping[target_key]
    backup_model = model_mapping["flagship"] or allowed_models[0]
    
    if primary_model == backup_model and len(allowed_models) > 1:
        backup_model = next((m for m in allowed_models if m != primary_model), allowed_models[0])
        
    # Structured formatting to guide models to output direct answer immediately
    user_prompt = f"Task: {prompt}\nFormat: Result only.\nAnswer:"
    
    # Try querying the local model first if category is local (0 Fireworks tokens)
    from src.router import should_route_local
    if should_route_local(category):
        try:
            sys.stderr.write(f"Routing task {task_id} ({category}) locally to save Fireworks tokens...\n")
            sys.stderr.flush()
            answer = query_local_model(user_prompt, sys_prompt, max_tokens, stop_sequences)
            return {"task_id": task_id, "answer": answer}
        except Exception as e:
            sys.stderr.write(f"Local model execution failed for task {task_id} ({category}): {e}. Falling back to Fireworks AI...\n")
            sys.stderr.flush()
            
    # Establish a dynamic, ordered fallback list of models to try (Defense B: Sequential degradation)
    candidates = [primary_model]
    if backup_model not in candidates:
        candidates.append(backup_model)
    for m in allowed_models:
        if m not in candidates:
            candidates.append(m)
            
    last_err = None
    for model in candidates:
        try:
            answer = request_with_retry(user_prompt, sys_prompt, model, api_key, base_url, max_tokens, stop_sequences)
            return {"task_id": task_id, "answer": answer}
        except Exception as e:
            sys.stderr.write(f"Model {model} failed for task {task_id} ({category}): {e}. Trying next fallback candidate...\n")
            sys.stderr.flush()
            last_err = e
            
    sys.stderr.write(f"All available models failed for task {task_id}. Using fallback answer. Last error: {last_err}\n")
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
