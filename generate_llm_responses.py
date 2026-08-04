import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic
from mistralai.client import Mistral

# Load keys from a local .env file (real environment variables win over it)
load_dotenv()

# 1. Dataset Path (override with the CONSUMER_COMPLAINTS_CSV env var)
DEFAULT_DATASET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "dataset", "consumer_complaints.csv"
)
dataset_path = os.getenv("CONSUMER_COMPLAINTS_CSV", DEFAULT_DATASET)


# 2. API keys come from the environment only -- never hardcode them in this file.
def require_key(name: str) -> str:
    key = os.getenv(name)
    if not key:
        raise SystemExit(
            f"Missing {name}. Set it before running, e.g. in PowerShell:\n"
            f'    $env:{name} = "your-key-here"'
        )
    return key


OPENAI_API_KEY = require_key("OPENAI_API_KEY")
ANTHROPIC_API_KEY = require_key("ANTHROPIC_API_KEY")
MISTRAL_API_KEY = require_key("MISTRAL_API_KEY")

# Initialize SDK Clients
openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
mistral_client = Mistral(api_key=MISTRAL_API_KEY)


# 3. Define Generator Functions
def get_chatgpt_response(prompt: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()


def get_claude_response(prompt: str) -> str:
    response = anthropic_client.messages.create(
        model="claude-sonnet-5",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def get_mistral_response(prompt: str) -> str:
    response = mistral_client.chat.complete(
        model="mistral-small-latest",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()


# 4. Load Dataset and Process First 5 Rows
df = pd.read_csv(dataset_path, low_memory=False)

# Handle column naming variations (e.g. 'Issue'/'issue', 'sub_issue'/'Sub-issue')
def find_col(*candidates):
    """Match a column ignoring case and any -, _ or space separators."""
    norm = {"".join(ch for ch in c.lower() if ch.isalnum()): c for c in df.columns}
    for cand in candidates:
        key = "".join(ch for ch in cand.lower() if ch.isalnum())
        if key in norm:
            return norm[key]
    raise KeyError(f"None of {candidates} found in {list(df.columns)}")


issue_col = find_col("issue")
sub_issue_col = find_col("sub_issue", "subissue")

sample_df = df.head(10)
results = []

print(f"--- Processing First 10 Issues using ChatGPT, Claude, and Mistral ---\n")

for idx, row in sample_df.iterrows():
    issue_text = str(row[issue_col]).strip()

    sub_issue_raw = row[sub_issue_col]
    if pd.isna(sub_issue_raw) or not str(sub_issue_raw).strip():
        sub_issue_text = ""
    else:
        sub_issue_text = str(sub_issue_raw).strip()
        if sub_issue_text.lower() == issue_text.lower():
            sub_issue_text = ""

    # Combined text handed to the LLMs
    if sub_issue_text:
        combined_text = f"Issue: {issue_text}\nSub-issue: {sub_issue_text}"
    else:
        combined_text = f"Issue: {issue_text}"

    # Prompt template for the LLMs
    prompt = (
        f"You are a customer service representative for a financial institution.\n"
        f"Briefly respond to the following consumer complaint in 2 sentences.\n"
        f"{combined_text}"
    )

    print(f"================ Row {idx + 1} ================")
    print(f"Issue: {issue_text}")
    print(f"Sub-issue: {sub_issue_text or '(none)'}\n")

    # Call ChatGPT
    try:
        gpt_out = get_chatgpt_response(prompt)
    except Exception as e:
        gpt_out = f"ChatGPT Error: {e}"

    # Call Claude
    try:
        claude_out = get_claude_response(prompt)
    except Exception as e:
        claude_out = f"Claude Error: {e}"

    # Call Mistral AI
    try:
        mistral_out = get_mistral_response(prompt)
    except Exception as e:
        mistral_out = f"Mistral Error: {e}"

    print(f"[ChatGPT]:\n{gpt_out}\n")
    print(f"[Claude]:\n{claude_out}\n")
    print(f"[Mistral]:\n{mistral_out}\n")
    print("-" * 50)

    results.append({
        "Row": idx + 1,
        "Issue": issue_text,
        "Sub_Issue": sub_issue_text,
        "ChatGPT": gpt_out,
        "Claude": claude_out,
        "Mistral": mistral_out,
    })

# Convert to DataFrame and save
output_df = pd.DataFrame(results)

output_path = os.path.join(os.path.dirname(dataset_path), "llm_responses_sample.csv")
output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
print(f"\nSaved {len(output_df)} rows x 3 models to:\n{output_path}")