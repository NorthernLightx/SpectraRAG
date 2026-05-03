"""Top-level src package.

Loads .env into os.environ as a side effect of importing anything under src/.
This is the single source of truth for dotenv loading: every entry point
(FastAPI app factory, eval/ingest scripts, tests, ad-hoc Python sessions)
imports something under src/, so .env is always loaded before the first
os.environ.get() call. Existing env vars win over .env (load_dotenv default
override=False), so CI / shell exports remain authoritative.

.env is gitignored (.gitignore: .env, .env.local, .env.*.local).
"""

from dotenv import load_dotenv

load_dotenv()
