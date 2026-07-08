# EcoRoute: Zero-Token Deterministic Multi-Model Routing Agent

EcoRoute is an enterprise-grade, lightweight, zero-token deterministic routing agent designed for Track 1 (General-Purpose AI Agent) of the AMD Developer Hackathon (ACT II). 

By implementing an advanced routing matrix based on deterministic heuristic parsing and token budget controls, EcoRoute achieves high accuracy across 8 standard capability categories while drastically minimizing inference token expenses.

## 🚀 Key Features

1. **Environment Interceptor**: Performs strict pre-execution checks on startup to validate critical Fireworks API environment variables, outputting detailed error streams and exiting gracefully with Code 10 on failure.
2. **JSON Input Parser**: A memory-efficient streaming scanner processing input tasks chunk by chunk (64KB memory footprint), scaling to extremely large task sizes without memory exhaustion.
3. **Catch-All Exception Shield**: Full-sandbox execution around each query with exponential backoff retries for rate-limiting (429) or backend (5xx) failures, automatically falling back to alternative models and placeholder safe outputs to ensure the pipeline always finishes with a successful `Exit Code 0`.
4. **Dynamic Heuristic Task Router**: Classifies blind-box natural language queries into 8 functional domains (Factual Q&A, Math, Logic, Sentiment, Summarization, NER, Code Gen, Code Debug) using lightweight string heuristic keyword matching.
5. **Cost-minimizing Model Selector**: Optimizes task routing between 5 permitted Fireworks models based on target complexity:
   - **Cheap tasks** (Sentiment, Summarisation, NER) -> Mapped to `gemma-4-26b-a4b-it`
   - **Mid-tier reasoning** (Math, Logic Puzzles) -> Mapped to `gemma-4-31b-it` / `gemma-4-31b-it-nvfp4`
   - **Coding tasks** (Debugging, Generation) -> Mapped to `kimi-k2p7-code`
   - **Flagship / Fallback** (Factual Q&A, Unknown queries) -> Mapped to `minimax-m3`
6. **Token Compression (Stage 3)**:
   - **Input-side cleaning**: Shrinks input strings by stripping leading/trailing whitespace, collapsing multiple tabs/spaces to a single space, and limiting consecutive newlines to maximum of 2.
   - **System prompt minimization**: Shortens category-specific instructions to just 3–5 words, saving significant system token overhead for bulk queries.
   - **Output token capping**: Enforces hard task-specific `max_tokens` request limits to stop models from generating verbose explanation filler.
7. **Atomic Compliant JSON Writer**: Guarantees zero-corruption of outputs by writing results to a temporary buffer file, calling `fsync()` to force disk flush, and performing atomic rename replacing via `os.replace` to prevent malformed outputs.

---

## 📂 Repository Structure

- [agent.py](file:///d:/Desktop/EcoRoute/agent.py): The main runtime agent pipeline.
- [test_agent.py](file:///d:/Desktop/EcoRoute/test_agent.py): Automated test suite.
- [requirements.txt](file:///d:/Desktop/EcoRoute/requirements.txt): Runtime pip packages.
- [Dockerfile](file:///d:/Desktop/EcoRoute/Dockerfile): Standard Docker setup.
- [Participant Guide...pdf](file:///d:/Desktop/EcoRoute/Participant%20Guide_%20AMD%20Developer%20Hackathon%20%28ACT%20II%29.pdf): PDF submission details.

---

## 🛠️ Local Development & Testing

### 1. Requirements Setup
```bash
pip install -r requirements.txt
```

### 2. Run Automated Verification Tests
We have built a test suite using `unittest` which verifies environment verification, streaming parsing, regex classification, model routing, exception catch/retrying, and atomic file replacements.
```bash
python test_agent.py
```

### 3. Run Pipeline Locally (Mock Data)
You can test the execution loop using local paths and mock credentials:
```bash
# Create local directories
mkdir input output

# Create mock tasks list
echo '[{"task_id": "t1", "prompt": "Identify the organization: Google was founded in 1998."}, {"task_id": "t2", "prompt": "Solve: 2 + 2 = ?"}]' > input/tasks.json

# Run agent with local paths overrides
$env:FIREWORKS_API_KEY="mock-key"
$env:FIREWORKS_BASE_URL="http://localhost:8000"
$env:ALLOWED_MODELS="minimax-m3,gemma-4-26b-a4b-it"
$env:INPUT_TASKS_PATH="input/tasks.json"
$env:OUTPUT_RESULTS_PATH="output/results.json"

python agent.py
```

---

## 🐳 Docker Deployment (Submission Instructions)

The judging VM runs `linux/amd64`. You must build and push your Docker container using the target platform manifest:

### 1. Build and Tag Image
```bash
docker buildx build --platform linux/amd64 --tag <your-dockerhub-username>/ecoroute-agent:latest --push .
```

### 2. Verify Image Footprint
The base image uses `python:3.12-slim-bookworm` to keep the compressed size below 150MB, guaranteeing it starts within milliseconds and easily conforms to the 10GB limit.
