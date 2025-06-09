import time

FAILED_LLM_LOG_FILE = "failed_llm_prompts_log.txt"


def read_file_content(file_path, default_content=""):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"[Warning] Prompt file not found: {file_path}.")
        return default_content
    except Exception as e:
        print(f"[Error] Reading prompt file {file_path}: {e}.")
        return default_content


def log_failed_llm_attempt(prompt, raw_response, error_type="Unknown Validation/Parse Error"):
    try:
        with open(FAILED_LLM_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"--- Failed LLM Attempt: {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            f.write(f"Error Type: {error_type}\n")
            f.write("Prompt Sent:\n")
            f.write(prompt + "\n\n")
            f.write("Raw LLM Response Received:\n")
            f.write(str(raw_response) + "\n")
            f.write("------------------------------------------------------------\n\n")
    except Exception as e:
        print(f"[ERROR] Could not write to failed LLM prompt log: {e}")
