# Setup Guide: LLM SDK & Development Environment

This project uses **Gemini as the primary, free API** for all executable
demos, since Google AI Studio issues a key with no credit card required
(~1,500 requests/day on gemini-2.5-flash). Correct OpenAI reference code
is also included and documented below for local use with your own billed
OpenAI account, since that was the API originally specified in the task.

---

## Part A: Gemini Setup (Free, No Credit Card — Recommended)

### Step 1: Get a Free Gemini API Key

1. Go to https://aistudio.google.com/apikey (sign in with any Google account).
2. Click "Create API key" -- no payment method, no credit card, no trial expiry.
3. Copy the key. Unlike OpenAI, this key can be viewed again later in AI Studio if needed, though treating it as a one-time-reveal secret is still good practice.

### Step 2: Configure as an Environment Variable

```
export GEMINI_API_KEY="your-real-key-here"
```

Verify it's set:
```
echo $GEMINI_API_KEY | cut -c1-6
```

### Step 3: Install the Gemini SDK

```
pip install google-genai
```

### Step 4: Run the Gemini Demo Scripts

```
python3 token_cost_calculator.py       # No API key needed -- pure arithmetic
python3 gemini_content_demo.py         # Needs GEMINI_API_KEY
python3 gemini_interactions_demo.py    # Needs GEMINI_API_KEY
python3 gemini_streaming_demo.py       # Needs GEMINI_API_KEY
```

**Free tier limits (gemini-2.5-flash):** roughly 15 requests/minute, 1,500 requests/day -- more than enough for this entire task with generous headroom.

---

## Part B: OpenAI Setup (Original Task Specification, Requires Billing)

Follow these steps in order if you specifically need to verify against OpenAI. Each one is a prerequisite for the next.

---

### Step 1: Register on the OpenAI Console and Generate an API Key

1. Go to platform.openai.com and sign up or log in.
2. Click your profile icon (top-right) -> "View API keys" (or navigate directly to platform.openai.com/api-keys).
3. Click "Create new secret key", give it a descriptive name (e.g., day8-sdk-setup), and click Create.
4. Copy the key immediately -- it is shown only once. If you lose it, delete the key and generate a new one; there is no way to view a lost key again.
5. Go to Billing (platform.openai.com/settings/billing) and add a payment method with a small amount of credit (a few dollars is more than enough for this entire task). Requests will fail with a 429 insufficient_quota error until billing credit exists, even with a valid key.

---

### Step 2: Configure the Key as an OS Environment Variable -- Never Hardcode Credentials

Why this matters: a hardcoded API key in source code gets committed to version control, potentially exposing it publicly in a GitHub repository forever (even if deleted in a later commit, it remains in git history). Environment variables keep secrets out of code entirely.

### On Linux/macOS (bash/zsh)

For the current terminal session only:
```
export OPENAI_API_KEY="sk-...your-real-key-here..."
```

To persist across terminal sessions, add the line above to ~/.bashrc (bash) or ~/.zshrc (zsh), then reload:
```
source ~/.bashrc
```

### On Windows (PowerShell)

```
$env:OPENAI_API_KEY = "sk-...your-real-key-here..."
```

To persist permanently, use System Properties -> Environment Variables, or:
```
[System.Environment]::SetEnvironmentVariable('OPENAI_API_KEY', 'sk-...your-real-key-here...', 'User')
```

### Verify it's set correctly (without printing the whole secret)

```
echo $OPENAI_API_KEY | cut -c1-10
```

This should show the start of your real key (e.g., sk-proj-A), confirming it's set without ever printing the full secret to your terminal history.

### Using python-dotenv as a project-local alternative

For project-specific configuration (recommended for this task), create a .env file in your project root:
```
OPENAI_API_KEY=sk-...your-real-key-here...
```

Critical: add .env to .gitignore immediately, before your first commit:
```
echo ".env" >> .gitignore
```

Then load it at the top of any script:
```python
from dotenv import load_dotenv
load_dotenv()  # reads .env and sets the variables into os.environ
```

---

### Step 3: Initialize an Isolated Virtual Environment

```
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate          # Windows
```

Your prompt should now show (venv) at the start.

---

### Step 4: Install Required Packages

```
pip install openai fastapi uvicorn python-dotenv
```

| Package | Purpose |
|---|---|
| openai | The official Python SDK for both Chat Completions and Responses APIs |
| fastapi | Production-grade web framework, used in later Phase 2 days to serve the chat backend |
| uvicorn | ASGI server that actually runs a FastAPI application |
| python-dotenv | Loads .env files into environment variables at runtime |

Verify the install:
```
python3 -c "import openai, fastapi, uvicorn, dotenv; print('All packages imported successfully')"
```

---

### Step 5: Run the Demo Scripts

```
python3 token_cost_calculator.py     # No API key needed -- pure arithmetic, verified logic
python3 chat_completions_demo.py     # Needs OPENAI_API_KEY + billing credit
python3 responses_api_demo.py        # Needs OPENAI_API_KEY + billing credit
python3 streaming_demo.py            # Needs OPENAI_API_KEY + billing credit
```

Expected behavior if OPENAI_API_KEY is not set: the script will fail immediately at client = OpenAI() with a clear "Missing credentials" error -- this is expected, correct behavior (fail fast and loud, rather than silently proceeding with no credentials).

Expected behavior if billing credit is not added: the script will fail with a 429 insufficient_quota error at the first actual API call -- the key itself is valid, but the account has no usable credit.

---

### Common Setup Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| Missing credentials | OPENAI_API_KEY not set in this terminal session | Re-run the export/$env: command; remember it doesn't persist across new terminal windows unless added to your shell profile |
| 429 insufficient_quota | Valid key, but no billing credit added | Go to platform.openai.com/settings/billing and add a payment method + credit |
| ModuleNotFoundError: No module named 'openai' | Virtual environment not activated, or package not installed inside it | Confirm (venv) shows in your prompt, then re-run pip install openai |
| Key accidentally committed to git | .env or a hardcoded key wasn't gitignored before the first commit | Immediately revoke the key on the OpenAI console and generate a new one -- a key visible in git history should always be treated as compromised, even if you remove it in a later commit |
