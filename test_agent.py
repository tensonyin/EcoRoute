import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# Add current directory to path to import agent
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agent

class TestAgentPipeline(unittest.TestCase):
    
    def setUp(self):
        # Create temp dir for files
        self.test_dir = tempfile.mkdtemp()
        self.input_file = os.path.join(self.test_dir, "tasks.json")
        self.output_file = os.path.join(self.test_dir, "results.json")
        
        # Save environment variables
        self.env_backup = dict(os.environ)
        
    def tearDown(self):
        # Restore environment variables
        os.environ.clear()
        os.environ.update(self.env_backup)
        
        # Clean up temp dir
        shutil.rmtree(self.test_dir)
        
    def test_environment_interceptor_success(self):
        os.environ["FIREWORKS_API_KEY"] = "test-key"
        os.environ["FIREWORKS_BASE_URL"] = "https://api.fireworks.ai"
        os.environ["ALLOWED_MODELS"] = "model-a, model-b"
        
        api_key, base_url, allowed_models = agent.check_env()
        self.assertEqual(api_key, "test-key")
        self.assertEqual(base_url, "https://api.fireworks.ai")
        self.assertEqual(allowed_models, ["model-a", "model-b"])

    def test_environment_interceptor_missing_key(self):
        os.environ.clear()
        os.environ["FIREWORKS_BASE_URL"] = "https://api.fireworks.ai"
        os.environ["ALLOWED_MODELS"] = "model-a"
        
        with self.assertRaises(SystemExit) as cm:
            agent.check_env()
        self.assertEqual(cm.exception.code, 10)

    def test_environment_interceptor_empty_models(self):
        os.environ["FIREWORKS_API_KEY"] = "test-key"
        os.environ["FIREWORKS_BASE_URL"] = "https://api.fireworks.ai"
        os.environ["ALLOWED_MODELS"] = " , "
        
        with self.assertRaises(SystemExit) as cm:
            agent.check_env()
        self.assertEqual(cm.exception.code, 10)

    def test_streaming_json_parser_valid(self):
        tasks_data = [
            {"task_id": "t1", "prompt": "Prompt 1"},
            {"task_id": "t2", "prompt": "Prompt with braces {abc} and \"quotes\""},
            {"task_id": "t3", "prompt": "Prompt with escaped \\\"quotes\\\" and \\\\ backslashes"}
        ]
        with open(self.input_file, "w", encoding="utf-8") as f:
            json.dump(tasks_data, f)
            
        tasks = list(agent.stream_tasks(self.input_file))
        self.assertEqual(len(tasks), 3)
        self.assertEqual(tasks[0]["task_id"], "t1")
        self.assertEqual(tasks[1]["task_id"], "t2")
        self.assertEqual(tasks[1]["prompt"], 'Prompt with braces {abc} and "quotes"')
        self.assertEqual(tasks[2]["task_id"], "t3")

    def test_streaming_json_parser_whitespace_and_newlines(self):
        tasks_str = """
        [
          {
            "task_id": "t1",
            "prompt": "Lines\\nand\\r\\nwhitespaces"
          },
          {
            "task_id": "t2",
            "prompt": "Another prompt"
          }
        ]
        """
        with open(self.input_file, "w", encoding="utf-8") as f:
            f.write(tasks_str)
            
        tasks = list(agent.stream_tasks(self.input_file))
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0]["task_id"], "t1")
        self.assertEqual(tasks[0]["prompt"], "Lines\nand\r\nwhitespaces")

    def test_clean_prompt(self):
        self.assertEqual(agent.clean_prompt("  hello   world  "), "hello world")
        self.assertEqual(agent.clean_prompt("line1\n\n\n\nline2"), "line1\n\nline2")
        self.assertEqual(agent.clean_prompt("hello\t\tworld"), "hello world")
        
        # Politeness removal
        self.assertEqual(agent.clean_prompt("Please help me to summarize the text: Hello"), "summarize: Hello")
        self.assertEqual(agent.clean_prompt("Could you please write a Python function to check primality"), "code to check primality")
        self.assertEqual(agent.clean_prompt("classify the sentiment of this text: 'Happy'"), "classify sentiment: 'Happy'")
        self.assertEqual(agent.clean_prompt("fix the runtime error in this code: print(x)"), "fix: print(x)")

    def test_classify_task(self):
        # Sentiment
        cat, sys_p, key, tokens = agent.classify_task("Please analyze the sentiment of this review: 'I hate it!'")
        self.assertEqual(cat, "sentiment")
        self.assertEqual(key, "cheap")
        self.assertEqual(tokens, 30)
        
        # Summarisation
        cat, sys_p, key, tokens = agent.classify_task("Please write a summary of the following essay")
        self.assertEqual(cat, "summarisation")
        self.assertEqual(key, "cheap")
        
        # NER
        cat, sys_p, key, tokens = agent.classify_task("Identify all organization entities in this sentence")
        self.assertEqual(cat, "ner")
        self.assertEqual(key, "cheap")
        
        # Code generation
        cat, sys_p, key, tokens = agent.classify_task("Write a python function to compute fibonacci numbers")
        self.assertEqual(cat, "code_generation")
        self.assertEqual(key, "code")
        
        # Code debug
        cat, sys_p, key, tokens = agent.classify_task("Fix the syntax error in the code below")
        self.assertEqual(cat, "code_debug")
        self.assertEqual(key, "code")
        
        # Math
        cat, sys_p, key, tokens = agent.classify_task("Calculate the probability of flipping two heads")
        self.assertEqual(cat, "math_reasoning")
        self.assertEqual(key, "mid_dense")
        
        # Logic
        cat, sys_p, key, tokens = agent.classify_task("Solve this logic puzzle: if Alice is telling the truth...")
        self.assertEqual(cat, "logic_puzzles")
        self.assertEqual(key, "mid_dense")
        
        # Factual QA
        cat, sys_p, key, tokens = agent.classify_task("What is the capital of France?")
        self.assertEqual(cat, "factual_qa")
        self.assertEqual(key, "flagship")

    def test_map_allowed_models_all_present(self):
        allowed = [
            "accounts/fireworks/models/minimax-m3",
            "kimi-k2p7-code",
            "gemma-4-31b-it",
            "gemma-4-26b-a4b-it"
        ]
        mapping = agent.map_allowed_models(allowed)
        self.assertEqual(mapping["cheap"], "gemma-4-26b-a4b-it")
        self.assertEqual(mapping["mid_dense"], "gemma-4-31b-it")
        self.assertEqual(mapping["code"], "kimi-k2p7-code")
        self.assertEqual(mapping["flagship"], "accounts/fireworks/models/minimax-m3")

    def test_map_allowed_models_fallbacks(self):
        allowed = ["minimax-m3"]
        mapping = agent.map_allowed_models(allowed)
        self.assertEqual(mapping["cheap"], "minimax-m3")
        self.assertEqual(mapping["mid_dense"], "minimax-m3")
        self.assertEqual(mapping["code"], "minimax-m3")
        self.assertEqual(mapping["flagship"], "minimax-m3")

    @patch('requests.post')
    def test_request_with_retry_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Expected model response"
                }
            }]
        }
        mock_post.return_value = mock_response
        
        res = agent.request_with_retry("hello", "sys_p", "model-a", "key", "https://api.fireworks.ai", 100, max_retries=1)
        self.assertEqual(res, "Expected model response")

    @patch('time.sleep', return_value=None)
    @patch('requests.post')
    def test_request_with_retry_rate_limit_and_success(self, mock_post, mock_sleep):
        mock_resp_429 = MagicMock()
        mock_resp_429.status_code = 429
        
        mock_resp_200 = MagicMock()
        mock_resp_200.status_code = 200
        mock_resp_200.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Success after retry"
                }
            }]
        }
        
        mock_post.side_effect = [mock_resp_429, mock_resp_200]
        
        res = agent.request_with_retry("hello", "sys_p", "model-a", "key", "https://api.fireworks.ai", 100, max_retries=2)
        self.assertEqual(res, "Success after retry")
        self.assertEqual(mock_post.call_count, 2)

    @patch('time.sleep', return_value=None)
    @patch('requests.post')
    def test_process_task_fallback_to_backup_model(self, mock_post, mock_sleep):
        mock_resp_err = MagicMock()
        mock_resp_err.status_code = 500
        
        mock_resp_ok = MagicMock()
        mock_resp_ok.status_code = 200
        mock_resp_ok.json.return_value = {
            "choices": [{
                "message": {
                    "content": "Backup model response"
                }
            }]
        }
        
        mock_post.side_effect = [mock_resp_err, mock_resp_err, mock_resp_err, mock_resp_ok]
        
        task = {"task_id": "t1", "prompt": "Hello"}
        model_mapping = {
            "cheap": "gemma-4-26b",
            "mid_dense": "gemma-4-31b",
            "mid_quant": "gemma-4-31b-nvfp4",
            "code": "kimi-k2p7-code",
            "flagship": "minimax-m3"
        }
        # Factual QA triggers flagship (minimax-m3), fallback triggers the next one
        result = agent.process_task(task, "key", "https://api.fireworks.ai", model_mapping, ["minimax-m3", "kimi-k2p7-code"])
        
        self.assertEqual(result["task_id"], "t1")
        self.assertEqual(result["answer"], "Backup model response")

    @patch('time.sleep', return_value=None)
    @patch('requests.post')
    def test_process_task_all_failed_uses_fallback_answer(self, mock_post, mock_sleep):
        mock_resp_err = MagicMock()
        mock_resp_err.status_code = 500
        mock_post.return_value = mock_resp_err
        
        task = {"task_id": "t1", "prompt": "Hello"}
        model_mapping = {
            "cheap": "gemma-4-26b",
            "mid_dense": "gemma-4-31b",
            "mid_quant": "gemma-4-31b-nvfp4",
            "code": "kimi-k2p7-code",
            "flagship": "minimax-m3"
        }
        result = agent.process_task(task, "key", "https://api.fireworks.ai", model_mapping, ["minimax-m3", "kimi-k2p7-code"])
        
        self.assertEqual(result["task_id"], "t1")
        self.assertEqual(result["answer"], "Failed to generate answer due to upstream error.")

    def test_atomic_write(self):
        results = [
            {"task_id": "t1", "answer": "Answer 1"},
            {"task_id": "t2", "answer": "Answer 2"}
        ]
        agent.write_results_atomic(results, self.output_file)
        self.assertTrue(os.path.exists(self.output_file))
        with open(self.output_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["task_id"], "t1")
        self.assertEqual(data[0]["answer"], "Answer 1")

if __name__ == "__main__":
    unittest.main()
