import os
import json
import requests
from policy import normalize_policy, validate_policy
from utils import FAILED_LLM_LOG_FILE, log_failed_llm_attempt, read_file_content

# These should be set by the main application
OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:!4b")
LLM_TIMEOUT = 240
LLM_TEMPERATURE_INIT_POP = 0.7
LLM_TEMPERATURE_MUTATION = 0.4

# --- Load Prompt File Paths from .env ---
POLICY_FORMAT_GUIDANCE_PATH = os.getenv("POLICY_FORMAT_GUIDANCE_PATH", "prompts/policy_format_guidance.md")
EXAMPLE_POLICY_PATH = os.getenv("EXAMPLE_POLICY_PATH", "prompts/example_policy.json")
INITIAL_POP_TEMPLATE_PATH = os.getenv("INITIAL_POP_TEMPLATE_PATH", "prompts/initial_population_template.md")
MUTATION_TEMPLATE_PATH = os.getenv("MUTATION_TEMPLATE_PATH", "prompts/intelligent_mutation_template.md")

LLM_PROMPT_POLICY_FORMAT_GUIDANCE = read_file_content(POLICY_FORMAT_GUIDANCE_PATH, "Error: Policy format guidance missing.")
LLM_EXAMPLE_POLICY_JSON_STR = read_file_content(EXAMPLE_POLICY_PATH, '{ "default": "flap" }')
LLM_INITIAL_POP_TEMPLATE = read_file_content(INITIAL_POP_TEMPLATE_PATH, "Error: Initial population template missing.")
LLM_MUTATION_TEMPLATE = read_file_content(MUTATION_TEMPLATE_PATH, "Error: Mutation template missing.")

OLLAMA_ENDPOINT = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:!4b")
LLM_TIMEOUT = 240; LLM_TEMPERATURE_INIT_POP = 0.7; LLM_TEMPERATURE_MUTATION = 0.4


def call_ollama_blocking(prompt, temperature):
    raw_response_text_for_log = "No response captured before error."
    try:
        payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=LLM_TIMEOUT)
        raw_response_text_for_log = response.text
        response.raise_for_status()
        response_dict = response.json()
        json_string_raw = response_dict.get('response', '')
        json_string = json_string_raw.split("```json")[1].strip()
        json_string = json_string.split("```")[0]
        if not json_string:
            log_failed_llm_attempt(prompt, raw_response_text_for_log, "LLM empty 'response' field")
            return None, raw_response_text_for_log
        parsed_json = json.loads(json_string)
        return parsed_json, json_string
    except requests.exceptions.Timeout:
        log_failed_llm_attempt(prompt, None, f"Timeout {LLM_TIMEOUT}s"); return None, "Request Timed Out"
    except requests.exceptions.RequestException as e:
        log_failed_llm_attempt(prompt,raw_response_text_for_log,f"Ollama connection error: {e}"); return None, f"Ollama Connection Error: {raw_response_text_for_log}"
    except json.JSONDecodeError as e:
        log_failed_llm_attempt(prompt,json_string if 'json_string' in locals() else raw_response_text_for_log,f"JSON decode error: {e}"); return None, json_string if 'json_string' in locals() else raw_response_text_for_log
    except Exception as e:
        log_failed_llm_attempt(prompt,raw_response_text_for_log,f"Unexpected error: {e}"); return None, f"Unexpected Error: {raw_response_text_for_log}"

def call_ollama_blocking_for_thread(prompt, temperature, result_container, error_container, raw_response_container):
    parsed_data, raw_text = call_ollama_blocking(prompt, temperature)
    raw_response_container['text'] = raw_text
    if parsed_data is not None:
        result_container['data'] = parsed_data
    else:
        error_container['error'] = f"LLM call failed. Raw: {(raw_text[:200] + '...') if raw_text else 'N/A'}"

def llm_generate_initial_population_prompt_str(n=5, failed_policies=None):
    failed_policies = failed_policies or []
    failed_json_str = json.dumps(failed_policies[-2:], indent=2) if failed_policies else "[]"
    prompt = LLM_INITIAL_POP_TEMPLATE.format(n=n, policy_format_guidance=LLM_PROMPT_POLICY_FORMAT_GUIDANCE,
        failed_json_str=failed_json_str, example_policy_json_str=LLM_EXAMPLE_POLICY_JSON_STR)
    return prompt

def llm_intelligent_mutation(policy):
    print(f"[LLM Call - Mutate]: Asking '{OLLAMA_MODEL}'...")
    policy_to_mutate_json_str = json.dumps(policy, indent=2)
    prompt = LLM_MUTATION_TEMPLATE.format(policy_to_mutate_json_str=policy_to_mutate_json_str,
        policy_format_guidance=LLM_PROMPT_POLICY_FORMAT_GUIDANCE, example_policy_json_str=LLM_EXAMPLE_POLICY_JSON_STR)
    parsed_data, raw_response = call_ollama_blocking(prompt, LLM_TEMPERATURE_MUTATION)
    if parsed_data is None:
        print(f"[LLM Failure - Mutate] Invalid/No response. See '{FAILED_LLM_LOG_FILE}'."); return None
    mutated_policy = normalize_policy(parsed_data)
    if mutated_policy and validate_policy(mutated_policy):
        print("[LLM Success - Mutate] Valid mutation."); return mutated_policy
    else:
        print(f"[LLM Failure - Mutate] Validation fail. Norm: {mutated_policy}. See '{FAILED_LLM_LOG_FILE}'.")
        log_failed_llm_attempt(prompt, raw_response or "No raw response", f"Validation failed. Norm: {mutated_policy}")
        return None
