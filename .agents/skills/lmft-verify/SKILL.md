---
name: lmft-verify
description: Run the LMFT-Architecture test suite, preflight checks, and regression verification under strict reproducibility rules.
---

# LMFT-Architecture Verification Protocol

Use this skill when checking working tree integrity, executing regression tests, or verifying that changes comply with the repository contracts.

## 1. Startup Order & Context Hygiene
Always verify context in this order before modifying or testing code:
1. `AGENTS.md` - working rules, anti-rigging rules, and context hygiene rules.
2. `PROJECT_STATE.md` - brief plain-English current status and next step (keep under 40 lines and 3,000 characters).
3. `DECISIONS.md` - brief plain-English log of the 5 most recent decisions (keep under 80 lines and 5,000 characters).

Do **not** read `LMFT-Architecture/docs/archive/` files at startup unless specifically needed.

## 2. Running Regression Tests
Execute in PowerShell from `Language Model Fuzzy Tree Research`:
```powershell
.\LMFT-Architecture\.venv\Scripts\python.exe -m pytest LMFT-Architecture/tests -q
```

To run a targeted test file:
```powershell
.\LMFT-Architecture\.venv\Scripts\python.exe -m pytest LMFT-Architecture/tests/<test_file>.py -q
```

## 3. Strict Verification Invariants
- **Frozen Seed:** Never modify, rename, reorder, or expand the 40 founding nodes in `LMFT-Architecture/docs/MINIMAL_LMFT_SPEC.md` (`LMFT-Architecture/src/lmft_architecture/birth.py`).
- **Zero Rigging:** Never hardcode test words into rubrics or move goalposts to make a failing test pass. Report failures (`0/4`, `2/5`, `FAILED`) plainly.
- **Plain English:** Keep all new code and documentation small, human-readable, and free of bureaucratic jargon.
