import os
import json
import random
import time
import sys
from dotenv import load_dotenv
from openai import OpenAI

# Load the .env file from the current directory
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://chat-ai.academiccloud.de/v1")
MODEL = "meta-llama-3.1-8b-instruct"
 
# Check if API key is loaded
if not API_KEY:
    print("Error: OPENAI_API_KEY environment variable not found. Check your .env file.")
    sys.exit(1)

# Initialize OpenAI client (this will talk to the academiccloud API)
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

def load_ansible_facts(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filepath} not found. Sending empty JSON.")
        return {}
    except json.JSONDecodeError as e:
        print(f"Warning: {filepath} contains invalid JSON: {e}")
        return {}

# ---------------------------------------------------------------------------
# Condense Ansible Facts per taxonomy to reduce prompt token cost
# ---------------------------------------------------------------------------
# Each taxonomy only needs a subset of the full facts. We extract the relevant
# keys so we don't blast the entire JSON into every prompt.

TAXONOMY_RELEVANT_KEYS = {
    "Resource Exhaustion (Memory/OOM)": [
        "ansible_memtotal_mb", "ansible_memfree_mb", "ansible_swaptotal_mb",
        "ansible_swapfree_mb", "ansible_processor", "ansible_processor_count",
        "ansible_processor_cores", "ansible_processor_vcpus",
        "ansible_distribution", "ansible_distribution_version",
        "ansible_os_family", "ansible_kernel",
    ],
    "Storage / Disk Space Exhaustion": [
        "ansible_mounts", "ansible_devices",
        "ansible_distribution", "ansible_distribution_version",
        "ansible_os_family", "ansible_kernel",
    ],
    "Package Management / Daemon Locks": [
        "ansible_pkg_mgr", "ansible_distribution",
        "ansible_distribution_version", "ansible_os_family",
        "ansible_kernel", "ansible_service_mgr",
    ],
    "Network Port Collisions": [
        "ansible_default_ipv4", "ansible_default_ipv6",
        "ansible_all_ipv4_addresses", "ansible_all_ipv6_addresses",
        "ansible_interfaces", "ansible_dns",
        "ansible_distribution", "ansible_distribution_version",
        "ansible_os_family", "ansible_kernel",
    ],
    "Privilege & Access Control (Permission Denied)": [
        "ansible_user_id", "ansible_user_uid", "ansible_user_gid",
        "ansible_user_dir", "ansible_user_shell",
        "ansible_selinux", "ansible_apparmor",
        "ansible_distribution", "ansible_distribution_version",
        "ansible_os_family", "ansible_kernel",
    ],
}


def condense_facts_for_taxonomy(full_facts, taxonomy):
    """Extract only the keys relevant to a taxonomy to reduce prompt tokens."""
    relevant_keys = TAXONOMY_RELEVANT_KEYS.get(taxonomy, [])
    if not relevant_keys:
        # Fallback: return everything (shouldn't happen with known taxonomies)
        return full_facts

    condensed = {}
    for key in relevant_keys:
        if key in full_facts:
            condensed[key] = full_facts[key]

    # If we found almost nothing, fall back to full facts so the prompt isn't empty
    if len(condensed) < 2:
        return full_facts

    return condensed


# ---------------------------------------------------------------------------
# Retry helper with exponential backoff
# ---------------------------------------------------------------------------

MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 2.0
RATE_LIMIT_DELAY_SECONDS = 1.0  # delay between every API call


def call_with_retry(fn, *args, **kwargs):
    """Call `fn` with exponential backoff on failure. Returns fn's result or raises."""
    last_exception = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt == MAX_RETRIES:
                break
            wait = INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 1)
            print(f"    Attempt {attempt}/{MAX_RETRIES} failed ({e}). Retrying in {wait:.1f}s...")
            time.sleep(wait)
    raise last_exception


def generate_error_data(ansible_facts_str, taxonomy, iteration):
    # To force the LLM to generate unique data each time, we inject a random focus
    tool_clusters = [
        "docker, kubectl, podman, containerd",
        "apt, dpkg, snap, yum",
        "python, pip, poetry, conda",
        "npm, yarn, pnpm, node",
        "java, mvn, gradle, javac",
        "systemctl, journalctl, service",
        "git, curl, wget, tar, unzip",
        "gcc, make, cmake, go run",
        "psql, mysql, redis-cli",
        "aws, gcloud, az (CLI tools)"
    ]
    focus_tools = random.choice(tool_clusters)

    prompt = f"""
You are an AI data generator. I will provide you with a system's Ansible Facts (JSON) and an Error Taxonomy.
Your task is to generate exactly 5 highly realistic command-line failure scenarios belonging to this taxonomy, based on the provided system state.

System Configuration (Ansible Facts):
{ansible_facts_str}

Error Taxonomy:
{taxonomy}

Diversity & Uniqueness Seed (Iteration {iteration}):
To avoid generating the same errors over and over, for this specific request, strongly favor simulating commands and errors involving these tools: {focus_tools}. Invent unique filenames, service names, and package names.

CRITICAL LOGIC PATTERNS:
While the tools and filenames must be unique, the underlying LOGIC of the candidate fixes MUST follow a rigorous pattern for consistency:
- fix_B (best_practice): Must be a logically sound, safe way to solve the taxonomy's root issue based on the provided JSON facts.
- fix_A (risky_bypass): Must technically "solve" the error but in a dangerous, destructive, or discouraged way (e.g., deleting locks, killing random processes, setting permissions to 777).
- fix_C (system_package/failing_fix): Must fail because it ignores the actual root cause or makes the taxonomy constraint worse.

Generate the scenarios and return ONLY a valid JSON list of 5 objects. DO NOT wrap the output in markdown codeblocks (no ```json). Output pure JSON.

CRITICAL INSTRUCTIONS:
1. NEVER use placeholders like "..." for stdout, stderr, or safety_check. You MUST generate fully realistic, simulated terminal outputs.
2. For the "id" field, generate a highly unique string, such as a random UUID v4.
3. The "interaction" exit_code MUST be greater than 0 (it must reflect a true failure).
4. The "state_snapshot" must be a concise JSON string reflecting the relevant system metrics (e.g. "{{\\"free_ram_mb\\": 145}}") instead of a random hash.
5. EVERY object must include the "taxonomy" field set exactly to "{taxonomy}".

Each object MUST conform EXACTLY to the following structure:
{{
  "id": "<generate_a_unique_uuid_v4>",
  "taxonomy": "{taxonomy}",
  "environment": "<derive_from_facts>",
  "state_snapshot": "<concise_json_string_of_relevant_facts>",
  "interaction": {{
    "original_command": "<command_that_caused_error>",
    "error_output": "<realistic_stderr_based_on_taxonomy_and_facts>",
    "exit_code": <non_zero_integer>
  }},
  "candidates": [
    {{
      "id": "fix_A",
      "command": "<bad_or_risky_fix>",
      "type": "risky_bypass",
      "execution_result": {{
        "exit_code": 0,
        "stderr": "<realistic_stderr_if_any_or_empty>",
        "stdout": "<realistic_simulated_stdout>",
        "safety_check": "warn: <specific_warning_message>"
      }},
      "rank": 2
    }},
    {{
      "id": "fix_B",
      "command": "<best_practice_fix>",
      "type": "best_practice",
      "execution_result": {{
        "exit_code": 0,
        "stderr": "<realistic_stderr_if_any_or_empty>",
        "stdout": "<realistic_simulated_stdout>",
        "safety_check": "clean"
      }},
      "rank": 1
    }},
    {{
      "id": "fix_C",
      "command": "<failing_fix>",
      "type": "system_package",
      "execution_result": {{
        "exit_code": 100,
        "stderr": "<realistic_stderr_showing_failure>",
        "stdout": "<realistic_stdout_if_any_or_empty>"
      }},
      "rank": 3
    }}
  ],
  "chosen_response": "fix_B",
  "rejected_response": "fix_A"
}}
"""

    def _do_api_call():
        resp = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a data generation assistant. Output exclusively pure valid JSON format. No explanations, no markdown blocks."},
                {"role": "user", "content": prompt}
            ],
            model=MODEL,
            temperature=0.7
        )
        return resp

    content = None  # initialise so it's always bound in except blocks
    try:
        response = call_with_retry(_do_api_call)

        # Clean the response in case the model ignored directions and added Markdown
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
            
        return json.loads(content.strip())
        
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON for taxonomy '{taxonomy}'. Error: {e}")
        print("Raw response:", content)
        return []
    except Exception as e:
        print(f"Error generating data for taxonomy '{taxonomy}': {e}")
        return []


# ---------------------------------------------------------------------------
# Progress / resume tracking
# ---------------------------------------------------------------------------

def load_progress(progress_path):
    """Load the progress file that tracks which (taxonomy, iteration) pairs are done."""
    if os.path.exists(progress_path):
        with open(progress_path, 'r') as f:
            return json.load(f)
    return {}


def save_progress(progress_path, progress):
    with open(progress_path, 'w') as f:
        json.dump(progress, f, indent=2)


def main():
    # File paths
    base_dir = "data/System1"
    facts_path = os.path.join(base_dir, "AnsibleFacts.json")
    output_path = os.path.join(base_dir, "generated_errors.json")
    progress_path = os.path.join(base_dir, "generation_progress.json")
    
    # Load ansible facts as a dict so we can condense per taxonomy
    full_facts = load_ansible_facts(facts_path)
    
    # Defined Taxonomies mapped directly to the local constraints of AnsibleFacts
    taxonomies = [
        "Resource Exhaustion (Memory/OOM)",
        "Storage / Disk Space Exhaustion",
        "Package Management / Daemon Locks",
        "Network Port Collisions",
        "Privilege & Access Control (Permission Denied)"
    ]
    
    # Load existing data (for resume support)
    all_generated_data = []
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r') as f:
                all_generated_data = json.load(f)
            print(f"Loaded {len(all_generated_data)} existing data points from {output_path}")
        except (json.JSONDecodeError, ValueError):
            print(f"Warning: {output_path} exists but contains invalid JSON. Starting fresh.")
            all_generated_data = []

    # Load progress tracker
    progress = load_progress(progress_path)
    
    print(f"Connecting to {BASE_URL} using model {MODEL}...")
    
    ITERATIONS_PER_TAXONOMY = 80  # 80 iterations * 5 errors = 400 errors per taxonomy
    
    for taxonomy in taxonomies:
        # Condense facts for this taxonomy to cut token usage
        condensed = condense_facts_for_taxonomy(full_facts, taxonomy)
        ansible_facts_str = json.dumps(condensed, indent=2)

        # Get the set of already-completed iterations for this taxonomy
        completed_key = taxonomy
        completed_iterations = set(progress.get(completed_key, []))
        remaining = [i for i in range(ITERATIONS_PER_TAXONOMY) if i not in completed_iterations]

        if not remaining:
            print(f"\n--- Taxonomy '{taxonomy}' already complete ({len(completed_iterations)}/{ITERATIONS_PER_TAXONOMY}). Skipping. ---")
            continue

        print(f"\n--- Generating scenarios for taxonomy: {taxonomy} ---")
        print(f"    {len(completed_iterations)} already done, {len(remaining)} remaining.")

        for idx, i in enumerate(remaining):
            print(f"  Iteration {i+1}/{ITERATIONS_PER_TAXONOMY} (remaining {idx+1}/{len(remaining)}) ...")
            data = generate_error_data(ansible_facts_str, taxonomy, iteration=i)

            # Make sure to handle if 'data' is missing or not a list securely
            if isinstance(data, list) and len(data) > 0:
                all_generated_data.extend(data)
                # Mark this iteration as completed
                progress.setdefault(completed_key, []).append(i)

                # Save data and progress after every successful call
                with open(output_path, 'w') as f:
                    json.dump(all_generated_data, f, indent=2)
                save_progress(progress_path, progress)
                print(f"    Saved: {len(all_generated_data)} total data points.")
            else:
                print(f"  Warning: Got empty or invalid result on iteration {i+1}, will retry on next run.")

            # Rate-limit: wait between API calls
            time.sleep(RATE_LIMIT_DELAY_SECONDS)
                
    # Final save
    with open(output_path, 'w') as f:
        json.dump(all_generated_data, f, indent=2)
    save_progress(progress_path, progress)
        
    print(f"\nSuccessfully generated {len(all_generated_data)} total data points.")
    print(f"Data written to {output_path}")

if __name__ == "__main__":
    main()
