# Equity Signal Backtester

## Start of every session
1. Read the project brain note in Obsidian:
   `/Users/gouthamrao/Documents/coding/Obsdian Vault/GVault/wiki/projects/Equity Signal Backtester.md`
   It holds current phase status, decisions, findings, open issues, and the work log. Don't ask Goutham for anything already recorded there.
2. Read `Skill.md` in this folder for the full spec (conventions, phases, tests, README rules).

## After every meaningful change
Update the brain note: phase status, key decisions, findings (real numbers only, with the source file), open issues, and append a dated Work log entry. Bump `updated` in its frontmatter. Also append one line to the vault's `Logs/YYYY-MM-DD.md` in the form `**HH:MM** - action | description`.

## Quick commands
- Env: `.venv/bin/python` (Python 3.11)
- Lint: `.venv/bin/ruff check .`
- Tests: `.venv/bin/pytest` (add `-m "not db"` if Postgres isn't running)
- DB: `docker compose up -d` (Postgres 16 on localhost:5433)
