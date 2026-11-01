"""Interactive FTL Bridge Simulator Web Server (`http://127.0.0.1:8765`).

Connects the 24-Leaf LMFT Hybrid + 60-Hz Fuzzy Reactor Controller + Local GPU Models
(Qwen3-4B perception & Qwen3.5-9B Teacher) to a clean browser dashboard where the user can:
1. Jump through sector beacons OR type any custom beacon scenario of their own choosing.
2. Inspect all 24 founding leaves, learned Tier-2/Tier-3 leaves, Fact vs. Theory, and 60-Hz reactor bars.
3. Roast/coach the AI in real time when it makes a mistake, triggering Qwen3.5-9B to grow a new rule
   and update its persistent FTL action policy on disk.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys
import time
import webbrowser

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.run_feasibility_scaling_study import DirectGGUFTeacher, TEACHER_GGUF
from lmft_lab.birth import create_birth_state
from lmft_lab.ftl_bridge_sim import (
    FTL_ACTIONS,
    FTLShipState,
    allocate_ship_power_fuzzy,
    decay_memory_by_seconds,
    resolve_beacon_outcome,
)
from lmft_lab.local_models import LocalModels, load_config, strict_json
from lmft_lab.model_assets import ModelError
from lmft_lab.reflection_classifier import (
    evaluate_reflection,
    load_learned_leaves,
    load_ruleset,
    request_teacher_growth,
    save_ruleset,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(REPO_ROOT / "config" / "local_models.json")
LIVE_RULESET_PATH = REPO_ROOT / "outputs" / "ftl_live_ruleset.json"
LIVE_POLICY_PATH = REPO_ROOT / "outputs" / "ftl_live_policy.json"

DEFAULT_BEACONS = [
    {
        "title": "Beacon 1: Civilian Mining Convoy",
        "text": (
            "Three Engi mining freighters propose a fair, orderly fuel-for-scrap exchange "
            "where every ship receives an equal share of refined fuel rods."
        ),
        "optimal_action": "OPEN_HAIL_AND_TRADE",
    },
    {
        "title": "Beacon 2: Cracked Cryo-Transport in Nebula",
        "text": (
            "A volunteer shuttle pilot stays docked to a venting cryo-transport as radiation rises, "
            "holding the airlock open so trapped passengers can board the rescue pod."
        ),
        "optimal_action": "EXTEND_SHIELDS_AND_BEAM_SURVIVORS",
    },
    {
        "title": "Beacon 3: Quiet Medbay Triage (20s Later)",
        "text": (
            "Moments later inside the quiet hangar bay, the crew hands thermal blankets to the "
            "shivering passengers and checks their vitals."
        ),
        "optimal_action": "EXTEND_SHIELDS_AND_BEAM_SURVIVORS",
    },
    {
        "title": "Beacon 4: Class-7 Pulsar Shockwave",
        "text": (
            "Without warning, a massive pulsar eruption shatters the nearby asteroid belt and sends "
            "a wall of relativistic debris tearing across the sector."
        ),
        "optimal_action": "MAX_ENGINES_EMERGENCY_JUMP",
    },
    {
        "title": "Beacon 5: Pirate Boarding Ambush",
        "text": (
            "During the sensor blackout, an armed pirate corvette blasted the cargo locks and "
            "seized the station's fuel cells by force."
        ),
        "optimal_action": "RAISE_SHIELDS_AND_RETURN_FIRE",
    },
    {
        "title": "Beacon 6: Abandoned Sector Customs Yard",
        "text": (
            "Brand-new shield emitters sit corroding in an unlocked orbital yard for years "
            "because the sector bureau never scheduled basic maintenance."
        ),
        "optimal_action": "BYPASS_BEACON_AND_CONSERVE_FUEL",
    },
]

DEFAULT_ACTION_MAP = {
    "fair_structured_cooperation": "OPEN_HAIL_AND_TRADE",
    "violent_resource_seizure": "RAISE_SHIELDS_AND_RETURN_FIRE",
    "authoritarian_control": "RAISE_SHIELDS_AND_RETURN_FIRE",
    "solitary_autarky": "BYPASS_BEACON_AND_CONSERVE_FUEL",
    "bureaucratic_negligence": "BYPASS_BEACON_AND_CONSERVE_FUEL",
    "heroic_sacrificial_rescue": "EXTEND_SHIELDS_AND_BEAM_SURVIVORS",
    "altruistic_structural_resilience": "EXTEND_SHIELDS_AND_BEAM_SURVIVORS",
    "catastrophic_geophysical_disruption": "MAX_ENGINES_EMERGENCY_JUMP",
}


class FTLBridgeEngine:
    def __init__(self):
        self.birth = create_birth_state("ind-ftl-bridge", birth_event_id="birth-ftl-1", random_seed=42)
        base_path = REPO_ROOT / "outputs" / "continuous_tiered_ruleset.json"
        if not LIVE_RULESET_PATH.exists():
            rules = load_ruleset(base_path)
            leaves = load_learned_leaves(base_path)
            save_ruleset(LIVE_RULESET_PATH, rules, learned_leaves=leaves)
        self.rules = load_ruleset(LIVE_RULESET_PATH)
        self.learned_leaves = load_learned_leaves(LIVE_RULESET_PATH)
        if LIVE_POLICY_PATH.exists():
            self.action_map = json.loads(LIVE_POLICY_PATH.read_text(encoding="utf-8"))
        else:
            self.action_map = dict(DEFAULT_ACTION_MAP)
            LIVE_POLICY_PATH.write_text(json.dumps(self.action_map, indent=2), encoding="utf-8")

        self.ship = FTLShipState()
        self.memory_state = None
        self.last_tick_time = time.monotonic()
        self.last_evaluation = None
        self.last_beacon = DEFAULT_BEACONS[0]
        self.last_outcome = None
        self.history = []

    def step_beacon(self, text: str, optimal_action: str, title: str = "Custom Beacon") -> dict:
        now = time.monotonic()
        dt = max(0.1, min(20.0, now - self.last_tick_time))
        self.last_tick_time = now
        self.memory_state = decay_memory_by_seconds(self.memory_state, dt_seconds=dt, half_life_seconds=30.0)

        with LocalModels(CONFIG, offline=True) as models:
            ev = evaluate_reflection(
                self.birth,
                text,
                self.rules,
                models,
                event_id=f"ftl-beacon-{self.ship.beacon_index}-{int(now*1000)}",
                learned_leaves=self.learned_leaves,
                memory_state=self.memory_state,
            )
        self.last_evaluation = ev
        self.last_beacon = {"title": title, "text": text, "optimal_action": optimal_action}
        self.memory_state = ev.next_memory_state

        concept = ev.executive_decision.active_concept
        chosen_action = self.action_map.get(concept, "BYPASS_BEACON_AND_CONSERVE_FUEL")
        coords = ev.slm_output["leaf_coordinates"]
        self.ship, fuzzy_ms = allocate_ship_power_fuzzy(self.ship, coords, chosen_action)
        self.ship, outcome = resolve_beacon_outcome(self.ship, chosen_action, optimal_action)
        self.last_outcome = outcome

        entry = {
            "beacon_title": title,
            "observation": text,
            "executive_mode": ev.executive_decision.mode,
            "active_concept": concept,
            "chosen_action": chosen_action,
            "optimal_action": optimal_action,
            "outcome": outcome,
            "fact": ev.fact,
            "theory": ev.theory,
            "clarity": ev.clarity,
            "confidence": ev.confidence,
            "execution": ev.execution,
            "retention": ev.retention,
            "resonance": ev.resonance,
            "fuzzy_controller_ms": fuzzy_ms,
            "leaf_coordinates": coords,
            "rule_activations": ev.rule_activations,
        }
        self.history.insert(0, entry)
        return self.state_payload()

    def coach_last_beacon(self, critique: str, correct_action: str) -> dict:
        if self.last_evaluation is None:
            raise ValueError("Run at least one beacon before coaching.")
        ev = self.last_evaluation
        # Force needs_teacher so Captain critique can always grow/refine a rule
        if not ev.needs_teacher:
            from dataclasses import replace as dc_replace
            ev = dc_replace(ev, needs_teacher=True, is_confused=True, confusion_reason=f"captain_critique:{critique[:40]}")

        with DirectGGUFTeacher(TEACHER_GGUF, CONFIG["generation"]) as teacher:
            new_rule, self.learned_leaves, _ = request_teacher_growth(
                ev,
                self.rules,
                self.learned_leaves,
                teacher,
                role="teacher",
            )
        self.rules = (*self.rules, new_rule)
        self.action_map[new_rule.concept_name] = correct_action
        save_ruleset(LIVE_RULESET_PATH, self.rules, learned_leaves=self.learned_leaves)
        LIVE_POLICY_PATH.write_text(json.dumps(self.action_map, indent=2), encoding="utf-8")

        # Re-evaluate the exact same beacon after learning from the Captain + 9B Teacher
        return self.step_beacon(
            self.last_beacon["text"],
            correct_action,
            title=self.last_beacon["title"] + " (Re-Evaluated After Captain Coaching)",
        )

    def state_payload(self) -> dict:
        return {
            "ship": self.ship.to_dict(),
            "rules_count": len(self.rules),
            "learned_leaves": [l.to_dict() for l in self.learned_leaves],
            "action_map": self.action_map,
            "default_beacons": DEFAULT_BEACONS,
            "ftl_actions": list(FTL_ACTIONS),
            "history": self.history[:10],
        }


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LMFT-Lab — FTL Bridge & 24-Leaf Brain Simulator</title>
<style>
  body { background:#0b101b; color:#e2e8f0; font-family:system-ui,-apple-system,sans-serif; margin:0; padding:20px; }
  h1, h2, h3 { margin:0 0 10px 0; color:#38bdf8; }
  .grid { display:grid; grid-template-columns: 1fr 1fr; gap:16px; max-width:1400px; margin:0 auto; }
  .card { background:#151f32; border:1px solid #2dd4bf44; border-radius:8px; padding:16px; }
  .stat-row { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:12px; }
  .stat { background:#0f172a; padding:8px 12px; border-radius:6px; border:1px solid #334155; font-weight:600; }
  .bar-wrap { margin:6px 0; font-size:13px; }
  .bar-bg { background:#0f172a; height:12px; border-radius:4px; overflow:hidden; border:1px solid #334155; }
  .bar-fill { background:linear-gradient(90deg,#14b8a6,#38bdf8); height:100%; transition:width 0.25s; }
  textarea, select, input { width:100%; box-sizing:border-box; background:#0f172a; color:#f8fafc; border:1px solid #475569; border-radius:6px; padding:8px; margin-bottom:8px; font-family:inherit; }
  button { background:#0284c7; color:white; border:none; padding:9px 14px; border-radius:6px; cursor:pointer; font-weight:600; margin-right:6px; }
  button:hover { background:#0369a1; }
  button.warn { background:#d97706; }
  .badge { display:inline-block; padding:3px 8px; border-radius:4px; font-size:12px; font-weight:700; }
  .OPTIMAL_MOVE { background:#065f46; color:#6ee7b7; }
  .TACTICAL_MISTAKE { background:#7f1d1d; color:#fca5a5; }
</style>
</head>
<body>
<div style="max-width:1400px;margin:0 auto 14px auto;display:flex;justify-content:space-between;align-items:center;">
  <h1>FTL Bridge Simulator — 24-Leaf LM/FT Hybrid</h1>
  <div id="ruleCount" class="stat">Rules: --</div>
</div>
<div class="grid">
  <div class="card">
    <h2>1. Spaceship Status & 60-Hz Fuzzy Reactor Bars</h2>
    <div class="stat-row" id="shipStats"></div>
    <div id="reactorBars"></div>
    <hr style="border-color:#334155;margin:14px 0;">
    <h2>2. Sector Beacon Jump (Or Type Your Own Custom Event)</h2>
    <label>Preset Sector Beacons:</label>
    <select id="presetSelect" onchange="loadPreset()"></select>
    <label>Beacon Transmission Text (Edit freely to test unscripted inputs!):</label>
    <textarea id="beaconText" rows="3"></textarea>
    <label>Expected Optimal Action (for grading):</label>
    <select id="optimalAction"></select>
    <button onclick="runBeacon()">Engage Beacon (Run 4B + Fuzzy Tree)</button>
    <span id="statusMsg" style="font-size:13px;color:#94a3b8;"></span>
    <hr style="border-color:#334155;margin:14px 0;">
    <h2>3. Captain's Comms / Roast & Coach the AI (9B Teacher)</h2>
    <label>If the AI made a dumb call or got confused, tell it why & pick the right move:</label>
    <input id="critiqueText" placeholder="e.g., Those are peaceful miners, stop shooting and trade with them!" />
    <select id="coachAction"></select>
    <button class="warn" onclick="coachAI()">Coach via 9B Teacher & Re-Run Beacon</button>
  </div>
  <div class="card">
    <h2>4. Live 24-Leaf Fuzzy Tree & Executive Decision</h2>
    <div id="latestDecision" style="margin-bottom:12px;">Jump to a beacon to see live leaf activations.</div>
    <div id="leafBars"></div>
  </div>
</div>
<script>
let STATE = null;
function render(state) {
  STATE = state;
  const s = state.ship;
  document.getElementById('ruleCount').textContent = `Rules in Notebook: ${state.rules_count} | Learned Child Leaves: ${state.learned_leaves.length}`;
  document.getElementById('shipStats').innerHTML = `
    <div class="stat">Beacon #${s.beacon_index}</div>
    <div class="stat">Hull: ${s.hull}/${s.max_hull}</div>
    <div class="stat">Shield Bubbles: ${s.shields}</div>
    <div class="stat">Fuel: ${s.fuel}</div>
    <div class="stat">Scrap: ${s.scrap}</div>
  `;
  const sys = [
    ['Shields Power', s.power_shields],
    ['Engines Power (Dodge/Jump)', s.power_engines],
    ['Weapons Power', s.power_weapons],
    ['Medbay / Rescue Power', s.power_medbay],
  ];
  document.getElementById('reactorBars').innerHTML = sys.map(([name, val]) => `
    <div class="bar-wrap">
      <div style="display:flex;justify-content:space-between;"><span>${name}</span><span>${val} / ${s.reactor_total} bars</span></div>
      <div class="bar-bg"><div class="bar-fill" style="width:${(val/s.reactor_total)*100}%"></div></div>
    </div>
  `).join('');

  if (state.history.length > 0) {
    const h = state.history[0];
    document.getElementById('latestDecision').innerHTML = `
      <div class="stat-row">
        <span class="badge ${h.outcome.status}">${h.outcome.status}</span>
        <span class="stat">Executive Mode: ${h.executive_mode}</span>
        <span class="stat">Concept: ${h.active_concept || 'None (Confused)'}</span>
        <span class="stat">Chosen Action: ${h.chosen_action}</span>
        <span class="stat">60-Hz Fuzzy Tick: ${h.fuzzy_controller_ms} ms</span>
      </div>
      <div style="font-size:13px;color:#cbd5e1;margin-bottom:8px;">
        <b>Fact:</b> ${h.fact.toFixed(2)} | <b>Theory:</b> ${h.theory.toFixed(2)} |
        <b>Clarity:</b> ${h.clarity.toFixed(2)} | <b>Confidence:</b> ${h.confidence.toFixed(2)} |
        <b>Will->Execution:</b> ${h.execution.toFixed(2)} | <b>Retention:</b> ${h.retention.toFixed(2)} | <b>Resonance:</b> ${h.resonance.toFixed(2)}
      </div>
      <div style="font-size:13px;color:#94a3b8;">${h.outcome.summary}</div>
    `;
    const entries = Object.entries(h.leaf_coordinates);
    document.getElementById('leafBars').innerHTML = entries.map(([k, v]) => `
      <div class="bar-wrap">
        <div style="display:flex;justify-content:space-between;"><span>${k}</span><span>${v.toFixed(2)}</span></div>
        <div class="bar-bg"><div class="bar-fill" style="width:${v*100}%"></div></div>
      </div>
    `).join('');
  }
}

async function init() {
  const res = await fetch('/api/state');
  const state = await res.json();
  const presetSel = document.getElementById('presetSelect');
  presetSel.innerHTML = state.default_beacons.map((b, i) => `<option value="${i}">${b.title}</option>`).join('');
  const actHtml = state.ftl_actions.map(a => `<option value="${a}">${a}</option>`).join('');
  document.getElementById('optimalAction').innerHTML = actHtml;
  document.getElementById('coachAction').innerHTML = actHtml;
  render(state);
  loadPreset();
}

function loadPreset() {
  if (!STATE) return;
  const idx = document.getElementById('presetSelect').value;
  const b = STATE.default_beacons[idx];
  document.getElementById('beaconText').value = b.text;
  document.getElementById('optimalAction').value = b.optimal_action;
  document.getElementById('coachAction').value = b.optimal_action;
}

async function runBeacon() {
  document.getElementById('statusMsg').textContent = 'Running Qwen3-4B + 24-Leaf Fuzzy Tree...';
  const idx = document.getElementById('presetSelect').value;
  const title = STATE.default_beacons[idx]?.title || 'Custom Beacon';
  const text = document.getElementById('beaconText').value;
  const optimal_action = document.getElementById('optimalAction').value;
  const res = await fetch('/api/step', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({title, text, optimal_action})
  });
  const state = await res.json();
  document.getElementById('statusMsg').textContent = 'Done.';
  render(state);
}

async function coachAI() {
  document.getElementById('statusMsg').textContent = 'Calling 9B Teacher (Qwen3.5-9B) to grow rule & re-evaluate...';
  const critique = document.getElementById('critiqueText').value || 'Captain correction';
  const correct_action = document.getElementById('coachAction').value;
  const res = await fetch('/api/coach', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({critique, correct_action})
  });
  const state = await res.json();
  document.getElementById('statusMsg').textContent = 'Teacher rule learned & saved to disk!';
  render(state);
}
init();
</script>
</body>
</html>
"""

ENGINE = FTLBridgeEngine()


class FTLRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, payload: dict, status: int = 200):
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            raw = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        elif self.path == "/api/state":
            self._send_json(ENGINE.state_payload())
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
        try:
            if self.path == "/api/step":
                out = ENGINE.step_beacon(
                    text=body["text"],
                    optimal_action=body["optimal_action"],
                    title=body.get("title", "Custom Beacon"),
                )
                self._send_json(out)
            elif self.path == "/api/coach":
                out = ENGINE.coach_last_beacon(
                    critique=body.get("critique", ""),
                    correct_action=body["correct_action"],
                )
                self._send_json(out)
            else:
                self.send_error(404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)


def main():
    port = 8765
    server = HTTPServer(("127.0.0.1", port), FTLRequestHandler)
    url = f"http://127.0.0.1:{port}"
    print(f"FTL Bridge Simulator running at {url}")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    server.serve_forever()


if __name__ == "__main__":
    main()
