---
name: lmft-verify
description: Run the LMFT-Lab test suite, preflight checks, and regression verification under strict reproducibility rules.
---

# LMFT-Lab Verification Protocol

Use this skill when checking working tree integrity, executing regression tests, or verifying that changes comply with the repository contracts.

## 1. Startup Order & Context Hygiene
Always verify context in this order before modifying or testing code:
1. `AGENTS.md` - working rules, anti-rigging rules, and context hygiene rules.
2. `PROJECT_STATE.md` - brief plain-English current status and next step (keep under 60 lines).
3. `DECISIONS.md` - brief plain-English log of current decisions.

Do **not** read `docs/archive/` files at startup unless specifically needed.

## 2. Running Regression Tests
The project requires `PYTHONPATH=src` for all test runs to ensure child worker processes resolve package modules correctly.

Execute in PowerShell from the repository root:
```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest -q
```

To run a targeted test file:
```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest tests/<test_file>.py -q
```

## 3. Strict Verification Invariants
- **Frozen Seed:** Never modify, rename, reorder, or expand the 40 founding nodes in `Minimal Specification Section 4`.
- **Zero Rigging:** Never hardcode test words into rubrics or move goalposts to make a failing test pass. Report failures (`0/4`, `2/5`, `FAILED`) plainly.
- **Plain English:** Keep all new code and documentation small, human-readable, and free of bureaucratic jargon.
