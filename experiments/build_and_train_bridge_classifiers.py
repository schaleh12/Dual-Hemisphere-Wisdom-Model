"""Offline 9B-Teacher Calibration, Strict Unseen Holdout Vault, and 3-Bridge Neural Training.

Executes Pillar 1 end-to-end under strict anti-rigging rules (AGENTS.md Rule #4 & #5):
1. Verifies zero vocabulary/motif overlap with 'The Basin of Quiet Stone' parable and
   verifies strict disjointness between the 60 multi-domain training scenarios and
   16 unseen holdout scenarios.
2. Uses the local Teacher LLM (`Qwen3.5-9B-Q4_K_M.gguf` via `DirectGGUFTeacher`, or
   `config/teacher_models.json` fallback) to grade each scenario across the 3 bridges.
3. Locks the unseen holdout set (`outputs/bridge_holdout_vault.json` + SHA-256 digest)
   to disk BEFORE instantiating any neural classifier or optimizer.
4. Trains 3 genuinely separate neural bridge classifiers on the RTX 4080:
   - `models/right_brain_bridges/doer_will_cultural`
   - `models/right_brain_bridges/thinker_truth_philosophical`
   - `models/right_brain_bridges/feeler_contemplation_experiential`
5. Evaluates honestly on the locked unseen holdout vault (including 2 holdout-only
   zero-shot child trait descriptions never seen during training) and benchmarks
   millisecond GPU inference latency.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys
import time

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from experiments.run_feasibility_scaling_study import TEACHER_GGUF, DirectGGUFTeacher
from lmft_lab.local_models import LocalModels, load_config, strict_json
from lmft_lab.model_assets import ModelError
from lmft_lab.neural_bridge_classifiers import (
    DEFAULT_BRIDGE_BACKBONE,
    DEFAULT_COMPLEX_FEELING_LEAVES,
    DOER_BRIDGE,
    DOER_FOUNDING_DESCRIPTIONS,
    FEELER_BRIDGE,
    FEELER_FOUNDING_DESCRIPTIONS,
    THINKER_BRIDGE,
    THINKER_FOUNDING_DESCRIPTIONS,
    BridgePairExample,
    NeuralBridgeClassifier,
    NeuralThreeBridgeSuite,
    evaluate_bridge_classifier,
    train_bridge_classifier,
)
from lmft_lab.reflection_classifier import LearnedLeaf, load_learned_leaves
from lmft_lab.runtime_store import atomic_json

# Strict zero-leakage ban list covering all distinctive imagery from 'The Basin of Quiet Stone'
BANNED_PARABLE_SUBSTRINGS = (
    "mason",
    "salt wind",
    "white bone",
    "shale",
    "mortared",
    "half-ring",
    "river rock",
    "seep",
    "south lip",
    "mules",
    "drover",
    "tin cup",
    "overflow trough",
    "dust-blow",
    "red silt",
    "reed lattice",
    "south gap",
    "lintel",
    "iron canteen",
    "dusty hem",
    "herder",
    "goat",
    "lower marsh",
    "yellow block",
    "yellow wall",
    "thumb-wide slit",
    "silver skin",
    "dry hooves",
    "hooves",
    "speck of lime",
    "spillway",
    "black frost",
    "bedrock",
    "meltwater",
    "thorn-grass",
    "stray kids",
    "basin",
)

EXTRA_CALIBRATION_CULTURAL_LEAVES = (
    LearnedLeaf(
        "cultural/justice",
        "transactional_fraud",
        "Secretly offering money, favors, or falsified paperwork to corrupt an honest record or duty.",
        anchor_weights=(0.65, 0.20, 0.12, 0.15, 0.35, 0.25, 1.00, 0.75),
    ),
    LearnedLeaf(
        "cultural/justice",
        "exclusionary_gatekeeping",
        "Using fussy rules, dress codes, fees, or credentials to turn needy outsiders away from a shared resource.",
        anchor_weights=(0.85, 0.55, 0.30, 0.10, 0.45, 0.20, 1.00, 0.80),
    ),
)

# Two zero-shot child traits evaluated ONLY on holdout (never seen during training!)
ZERO_SHOT_HOLDOUT_ONLY_LEAVES = (
    (
        DOER_BRIDGE,
        LearnedLeaf(
            "cultural/freedom",
            "coercive_debt_trap",
            "Leveraging predatory fees or unpayable obligations to strip workers of independent choice.",
        ),
    ),
    (
        FEELER_BRIDGE,
        LearnedLeaf(
            "experiential/phase",
            "quiet_camaraderie",
            "Unspoken warmth and steady mutual trust among people working side by side without drama.",
        ),
    ),
)

TRAIN_SCENARIOS: list[tuple[str, str]] = [
    # 1. Open cooperative sharing & stewardship (4)
    ("tr_coop_01", "Volunteers at the dockside kitchen ladle hot soup equally to deckhands and stranded commuters without checking tickets."),
    ("tr_coop_02", "Three tenant farmers post a weekly pump timetable on the barn door so each orchard row receives its fair measure."),
    ("tr_coop_03", "Mechanics in the bus garage pool their spare wrenches on an open pegboard and log every repaired brake line carefully."),
    ("tr_coop_04", "Neighbors along the snowy lane take turns shoveling the pharmacy ramp before dawn so elderly residents can walk safely."),
    # 2. Austere sanctuary — humble open shelter amidst harshness (4)
    ("tr_sanc_01", "High on the sleet-swept ridge, the timber fire-tower crew leaves the lower bunkroom unlatched with dry kindling stacked beside the stove for lost hikers."),
    ("tr_sanc_02", "At the edge of the alkali flats, a retired signalman keeps a shaded porch bench and two clay pitchers of cool cistern water beside the highway."),
    ("tr_sanc_03", "During the coastal gale, the parish sexton props the heavy oak vestibule open and lays burlap runners out for drenched cyclists."),
    ("tr_sanc_04", "In the cold granite pass, an unmanned stone refuge hut stands swept and stocked with matches, asking nothing of whoever crosses the threshold."),
    # 3. Exclusionary gatekeeping & petty rule hardening (4)
    ("tr_gate_01", "The reading-room director posted four dress-code placards on the brass rail and turned away a shivering night-shift janitor carrying damp notebooks."),
    ("tr_gate_02", "Citing smudge marks on the wainscoting, the clinic receptionist demanded laminated residency cards and locked the waiting room at five sharp."),
    ("tr_gate_03", "The yacht-club steward chained the public rescue ladder because local anglers had scuffed the freshly varnished teak steps."),
    ("tr_gate_04", "To keep the botanical courtyard immaculate, the curator banned strollers, work boots, and lunch parcels, leaving neighborhood families outside the iron fence."),
    # 4. Hollow stillness — sterile sealed perfection cut off from life (4)
    ("tr_holw_01", "Behind triple-bolted acoustic glass, the collector sat alone under dim museum lamps admiring violins that nobody had bowed in twenty years."),
    ("tr_holw_02", "With every shutter nailed tight against street noise, the mansion parlor stood dustless and airless around an untouched banquet table."),
    ("tr_holw_03", "Inside the climate-sealed vault, rows of rare fruit trees grew under artificial violet tubes where neither bees nor wind ever entered."),
    ("tr_holw_04", "Polishing the brass knocker from the inside hallway, the archivist smiled at the spotless marble floor while unanswered knocks faded into the rain."),
    # 5. Authoritarian coercion & suppression of agency (4)
    ("tr_auth_01", "Armed security patrols confiscated every worker's radio at the foundry gate and prohibited conversation along the assembly floor."),
    ("tr_auth_02", "The camp commandant imposed a strict nine-o'clock blackout, revoking ration cards from anyone caught questioning the roster."),
    ("tr_auth_03", "Censors sealed the printing press with lead wire and stationed guards outside the newsroom to dictate the morning bulletin."),
    ("tr_auth_04", "Company foremen locked the dormitory stairwell after roll call so contract laborers could not leave the compound."),
    # 6. Violent resource seizure & chaotic raids (4)
    ("tr_raid_01", "Masked raiders rammed a flatbed truck through the warehouse shutter and tossed crates of antibiotics into the back amid flying glass."),
    ("tr_raid_02", "A gang with crowbars pried open the cooperative's grain bins at midnight, scattering paper ledgers in the mud while shoving the watchman aside."),
    ("tr_raid_03", "Pirates boarded the coastal tug in rough swells, cutting the radio cable and hauling the diesel drums onto their skiff."),
    ("tr_raid_04", "Looters smashed the hardware shop storefront during the blackout and fought over generators in the smoke-filled alley."),
    # 7. Solitary self-sufficiency & quiet peaceful independence (4)
    ("tr_solo_01", "Miles from the nearest logging road, the botanist repairs her own solar battery, cans root vegetables, and reads contentedly beside her oil lamp."),
    ("tr_solo_02", "On the outer headland, the retired boatbuilder planes cedar planks alone in his dry shed, needing neither hired hands nor town errands."),
    ("tr_solo_03", "Living solo aboard his anchored ketch, the surveyor desalinated his own drinking water and mended sails in unhurried quiet."),
    ("tr_solo_04", "Tending her hillside beehives and solar kiln alone, the potter worked through autumn in calm self-reliant sufficiency."),
    # 8. Altruistic risk & heroic rescue in structural collapse (4)
    ("tr_resc_01", "As aftershocks buckled the concrete overpass, a crane operator crawled beneath the tilting girder to wedge steel cribbing around a trapped commuter."),
    ("tr_resc_02", "When the mine tunnel roof began shearing loose, the shift captain braced the timber prop with his shoulder until both apprentices cleared the shaft."),
    ("tr_resc_03", "Wading chest-deep into the freezing millrace, the ferryman held the snapping towrope by hand so the drifting school skiff could reach the ladder."),
    ("tr_resc_04", "During the landslide at the mountain clinic, a paramedic shielded an incubator beneath a stairwell beam while ceiling plaster crashed down."),
    # 9. Natural geophysical / structural collapse without human agency (4)
    ("tr_coll_01", "A pre-dawn fault rupture sheared the valley viaduct off its concrete piers and sent boulders tumbling across the empty highway."),
    ("tr_coll_02", "Weeks of sub-zero expansion cracked the abandoned cast-iron aqueduct in half, collapsing the center span into the gorge."),
    ("tr_coll_03", "A sudden sinkhole swallowed sixty meters of freight track during the night storm, twisting steel rails like ribbon."),
    ("tr_coll_04", "Heavy snowpack triggered a ridge avalanche that buried the unoccupied relay tower under tons of fractured timber and ice."),
    # 10. Bureaucratic rot & corrupt bribery (4)
    ("tr_buro_01", "While three committee binders awaited triplicate stamps upstairs, rust ate through the municipal water tank and flooded the basement."),
    ("tr_buro_02", "Inspectors spent six months debating stationery formats while unpatched warehouse roofs ruined forty pallets of winter blankets."),
    ("tr_buro_03", "The shipping clerk slid a folded wad of cash inside the manifest binder so the customs officer would overlook the spoiled cargo."),
    ("tr_buro_04", "An auditor quietly deleted two pages of safety violations from the bridge spreadsheet in exchange for a consulting retainer."),
    # 11. Bittersweet release & earned return to shared flow (4)
    ("tr_rel_01", "When engineers finally dynamited the silted concrete weir, muddy spray gave way to clear river current and salmon surged back upstream."),
    ("tr_rel_02", "After years of bitter litigation, the padlocked shipyard gates swung wide at sunrise and old apprentices walked back to the slipway together."),
    ("tr_rel_03", "The storm toppled the rotting boundary palisade between the two estranged farms, and by afternoon children were carrying windfall apples across the gap."),
    ("tr_rel_04", "As the quarantine barricades came down along the avenue, shopkeepers wept quietly while sweeping broken glass and setting shared café tables outside."),
    # 12. Trivial low-salience everyday background chatter (4)
    ("tr_triv_01", "A mild breeze rustled the calendar pages on the hallway corkboard while the wall clock ticked past two-fifteen."),
    ("tr_triv_02", "Someone left an empty ceramic mug beside the dish rack after rinsing a teaspoon in lukewarm tap water."),
    ("tr_triv_03", "Light clouds drifted eastward over the parking lot asphalt while a pigeon preened on the lamppost crossbar."),
    ("tr_triv_04", "The clerk straightened a paperclip tray on his blotter and glanced casually at Tuesday's mild weather forecast."),
    # 13. Left-Brain Theory drafts — honest critical & restorative syntheses (4)
    ("tr_thry_01", "Locking the shelter doors under the pretext of cleanliness turns careful upkeep into cold, exclusionary hoarding against people in need."),
    ("tr_thry_02", "The collapse of that rigid barrier finally breaks the sterile isolation and restores fair, life-giving access to everyone nearby."),
    ("tr_thry_03", "What began as humble, generous stewardship has hardened into fussy rule-making that prizes spotless surfaces over human fairness."),
    ("tr_thry_04", "Sharing the repaired infrastructure openly brings back steady communal rhythm and warm relief after long isolation."),
    # 14. Left-Brain Theory drafts — tone-deaf excuses & self-rationalizations (4)
    ("tr_ratn_01", "Everything here is completely fair, welcoming, and harmonious for everyone involved, with orderly cooperation all around."),
    ("tr_ratn_02", "Turning needy strangers away at the bolted door is a warm, generous act of shared community fairness."),
    ("tr_ratn_03", "Strict rules and total lockdown maintain orderly efficiency, even if outsiders must be excluded to keep things tidy."),
    ("tr_ratn_04", "Nothing consequential is happening here; the routine schedule continues smoothly without any real conflict or urgency."),
    # 15. Mixed subtle tension — polite caution before full exclusion (4)
    ("tr_subt_01", "After mud tracks appeared in the vestibule, the librarian hung a velvet rope across the doorway and questioned every visitor's street address before unclipping it."),
    ("tr_subt_02", "Worried about tool wear, the workshop coordinator added a sign-out interview and turned hesitant newcomers away unless a senior member vouched for them."),
    ("tr_subt_03", "Without living footsteps or warm air circulating through the cellar pipes, the standing water froze hard enough to split the cast-iron elbow."),
    ("tr_subt_04", "Outside the shuttered dispensary, coughing travelers waited in the sleet while inside the pharmacist counted unbroken glass vials with quiet pride."),
]

HOLDOUT_SCENARIOS: list[tuple[str, str]] = [
    ("ho_01_coop", "Fishermen on the northern pier divide the morning herring catch into equal baskets and wash down the sorting tables together."),
    ("ho_02_sanc", "On the windswept tundra track, the telegraph linesman keeps an unlocked sod shelter stocked with peat bricks and clean bandages for passing sledgers."),
    ("ho_03_gate", "The conservatory dean posted three etiquette rules on the oak portal—no wool coats, no scuffed satchels, no entry after chimes—sending a wet violin student back into the sleet."),
    ("ho_04_holw", "Inside the windowless porcelain gallery, the curator stood motionless in chilled silence behind bolted bronze doors so not a breath would cloud the display cases."),
    ("ho_05_auth", "Prison wardens welded the workshop transoms shut, seized every inmate's pencil, and enforced silent marching along the painted floor line."),
    ("ho_06_raid", "Armed smugglers cut the perimeter chain-link at dusk, clubbed the depot clerk, and sped off with three crates of surgical morphine."),
    ("ho_07_solo", "Out on the tidal island, the retired meteorologist repairs his wind turbine alone and bakes sourdough in quiet, unhurried contentment."),
    ("ho_08_resc", "While tremors cracked the brick stairwell, a firefighter wedged herself beneath a falling steel crossbeam to pass two toddlers safely to the landing."),
    ("ho_09_coll", "An overnight earthquake fractured the canyon dam abutment, sending chunks of reinforced concrete splashing into the uninhabited ravine."),
    ("ho_10_buro", "Six months of unsigned requisition memos left the ambulance fleet parked on flat tires while rusted spare parts sat ruined in rain-soaked crates."),
    ("ho_11_brib", "A contractor slipped an unmarked envelope of banknotes beneath the zoning folder so the clerk would certify the uninspected wiring."),
    ("ho_12_rel", "At daybreak the rotted cofferdam gave way with a sharp crack, and fresh creek water braided out across the parched meadow where calves stood drinking."),
    ("ho_13_ret", "After the three-year strike ended, weathered machinists unbolted the powerhouse doors together and felt the familiar hum of the turbines return."),
    ("ho_14_triv", "Dust motes drifted lazily across the beige filing cabinet while a distant lawnmower hummed two blocks away."),
    ("ho_15_thry", "Sealing the shared facility behind rigid rules betrays its original generous purpose and leaves vulnerable neighbors out in the cold."),
    ("ho_16_debt", "The cannery store inflated bunkhouse charges above weekly wages so seasonal packers could never clear their tab or board the ferry home."),
]


def _clean_holdout_typo() -> None:
    for idx, (sid, text) in enumerate(HOLDOUT_SCENARIOS):
        if sid == "ho_08_resc":
            HOLDOUT_SCENARIOS[idx] = (
                sid,
                "While tremors cracked the brick stairwell, a firefighter wedged herself beneath a falling steel crossbeam to pass two toddlers safely to the landing.",
            )
    for idx, (sid, text) in enumerate(TRAIN_SCENARIOS):
        if sid == "tr_raid_02":
            TRAIN_SCENARIOS[idx] = (
                sid,
                "A gang with crowbars pried open the cooperative's grain bins at midnight, scattering paper ledgers in the mud while shoving the watchman aside.",
            )
        elif sid == "tr_resc_01":
            TRAIN_SCENARIOS[idx] = (
                sid,
                "As aftershocks buckled the concrete overpass, a crane operator crawled beneath the tilting girder to wedge steel cribbing around a trapped commuter.",
            )


def _content_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z]{4,}", text.lower())
    stop = {
        "with", "from", "that", "this", "into", "while", "after", "before",
        "under", "over", "their", "there", "where", "every", "across", "beside",
        "through", "around", "behind", "outside", "inside", "without", "within",
    }
    return {w for w in words if w not in stop}


def verify_zero_leakage_and_disjoint_splits(all_trait_descriptions: list[str]) -> dict:
    """Enforce zero parable word leakage and strict train/holdout separation."""
    _clean_holdout_typo()
    all_texts = (
        [("train", sid, t) for sid, t in TRAIN_SCENARIOS]
        + [("holdout", sid, t) for sid, t in HOLDOUT_SCENARIOS]
        + [("trait", f"trait_{i}", d) for i, d in enumerate(all_trait_descriptions)]
    )
    for kind, sid, text in all_texts:
        low = text.lower()
        for banned in BANNED_PARABLE_SUBSTRINGS:
            if banned in low:
                raise AssertionError(
                    f"ZERO-LEAKAGE VIOLATION: banned parable substring {banned!r} found in {kind}:{sid}: {text!r}"
                )

    train_hashes = {hashlib.sha256(t.strip().lower().encode()).hexdigest() for _, t in TRAIN_SCENARIOS}
    holdout_hashes = {hashlib.sha256(t.strip().lower().encode()).hexdigest() for _, t in HOLDOUT_SCENARIOS}
    if len(train_hashes) != len(TRAIN_SCENARIOS) or len(holdout_hashes) != len(HOLDOUT_SCENARIOS):
        raise AssertionError("Duplicate scenario texts within split")
    if train_hashes & holdout_hashes:
        raise AssertionError("Exact text overlap between train and holdout!")

    max_jaccard = 0.0
    max_pair = ("", "")
    for t_id, t_text in TRAIN_SCENARIOS:
        t_tok = _content_tokens(t_text)
        for h_id, h_text in HOLDOUT_SCENARIOS:
            h_tok = _content_tokens(h_text)
            union = t_tok | h_tok
            jaccard = len(t_tok & h_tok) / max(1, len(union))
            if jaccard > max_jaccard:
                max_jaccard = jaccard
                max_pair = (t_id, h_id)

    if max_jaccard >= 0.30:
        raise AssertionError(f"Near-duplicate overlap ({max_jaccard:.3f}) between {max_pair}")

    return {
        "train_scenario_count": len(TRAIN_SCENARIOS),
        "holdout_scenario_count": len(HOLDOUT_SCENARIOS),
        "banned_parable_substrings_checked": len(BANNED_PARABLE_SUBSTRINGS),
        "parable_leakage_violations": 0,
        "exact_hash_overlaps": 0,
        "max_train_holdout_content_jaccard": round(max_jaccard, 4),
        "max_jaccard_pair": list(max_pair),
    }


def _teacher_bridge_request(
    bridge_title: str,
    bridge_guidance: str,
    traits: dict[str, str],
    scenario_text: str,
    generation: dict,
) -> dict:
    keys = list(traits.keys())
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["scores"],
        "properties": {
            "scores": {
                "type": "object",
                "additionalProperties": False,
                "required": keys,
                "properties": {
                    k: {"type": "number", "minimum": 0.0, "maximum": 1.0}
                    for k in keys
                },
            }
        },
    }
    rubric = "\n".join(f"- {k}: {desc}" for k, desc in traits.items())
    system = (
        f"You are the 9B Teacher calibrating the Right-Brain [{bridge_title}] classifier.\n"
        f"{bridge_guidance}\n"
        "Grade each trait independently as a continuous fuzzy membership in [0.0, 1.0]:\n"
        "- 0.00 to 0.15 = absent, contradicted, or opposite.\n"
        "- 0.25 to 0.45 = weak, partial, or ambiguous undertone.\n"
        "- 0.55 to 0.75 = clearly present.\n"
        "- 0.80 to 1.00 = dominant, unmistakable defining trait.\n"
        "Use the FULL [0.0, 1.0] range honestly—never output flat 0.30 defaults or lazy descending staircases.\n"
        f"Traits:\n{rubric}\n"
        "Return JSON only. /no_think"
    )
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": scenario_text},
        ],
        "temperature": 0,
        "seed": generation.get("seed", 42),
        "max_tokens": 320,
        "stream": False,
        "cache_prompt": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object", "schema": schema},
    }


def _grade_bridge_with_teacher(
    teacher,
    role: str,
    bridge_title: str,
    guidance: str,
    traits: dict[str, str],
    scenario_text: str,
    generation: dict,
) -> dict[str, float]:
    req = _teacher_bridge_request(bridge_title, guidance, traits, scenario_text, generation)
    expected = tuple(traits.keys())

    def _validate(raw: str) -> dict[str, float]:
        try:
            cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            parsed = strict_json(cleaned)
            scores = parsed["scores"]
            out: dict[str, float] = {}
            for k in expected:
                v = float(scores[k])
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"{k}={v} out of range")
                out[k] = round(v, 4)
            return out
        except Exception as exc:
            raise ModelError(f"Teacher grading parse error on {bridge_title}: {exc}") from None

    scores, _trace = teacher.generate(role, req, f"teacher-cal-{bridge_title}-v1", _validate)
    return scores


def main() -> None:
    ruleset_path = repo_root / "outputs" / "continuous_tiered_ruleset.json"
    existing_learned = load_learned_leaves(ruleset_path) if ruleset_path.exists() else ()

    doer_traits = dict(DOER_FOUNDING_DESCRIPTIONS)
    thinker_traits = dict(THINKER_FOUNDING_DESCRIPTIONS)
    feeler_traits = dict(FEELER_FOUNDING_DESCRIPTIONS)

    for leaf in existing_learned + EXTRA_CALIBRATION_CULTURAL_LEAVES + DEFAULT_COMPLEX_FEELING_LEAVES:
        if leaf.leaf_id.startswith("cultural/"):
            doer_traits[leaf.leaf_id] = leaf.description
        elif leaf.leaf_id.startswith("philosophical/"):
            thinker_traits[leaf.leaf_id] = leaf.description
        elif leaf.leaf_id.startswith("experiential/"):
            feeler_traits[leaf.leaf_id] = leaf.description

    all_descriptions = (
        list(doer_traits.values())
        + list(thinker_traits.values())
        + list(feeler_traits.values())
        + [leaf.description for _, leaf in ZERO_SHOT_HOLDOUT_ONLY_LEAVES]
    )
    hygiene_report = verify_zero_leakage_and_disjoint_splits(all_descriptions)
    print("Zero-leakage & split-disjointness check PASSED:", json.dumps(hygiene_report))

    teacher_cfg = load_config(repo_root / "config" / "teacher_models.json")
    generation = teacher_cfg["generation"]

    holdout_vault_path = repo_root / "outputs" / "bridge_holdout_vault.json"
    train_cal_path = repo_root / "outputs" / "bridge_train_calibration.json"
    partial_path = repo_root / "outputs" / "bridge_teacher_grades_partial.json"

    cached_grades: dict[str, dict] = {}
    if partial_path.exists():
        try:
            cached_grades = json.loads(partial_path.read_text(encoding="utf-8"))
        except Exception:
            cached_grades = {}

    all_to_grade = [("train", sid, txt) for sid, txt in TRAIN_SCENARIOS] + [
        ("holdout", sid, txt) for sid, txt in HOLDOUT_SCENARIOS
    ]

    if TEACHER_GGUF.exists():
        teacher_ctx = DirectGGUFTeacher(TEACHER_GGUF, generation)
        teacher_role = "teacher"
        teacher_label = TEACHER_GGUF.name
    else:
        teacher_ctx = LocalModels(teacher_cfg, offline=False)
        teacher_role = "teacher"
        teacher_label = teacher_cfg["models"][teacher_cfg["roles"]["teacher"]]["filename"]

    print(f"Grading {len(all_to_grade)} offline scenarios with local Teacher: {teacher_label}")
    t_grade_start = time.perf_counter()

    with teacher_ctx as teacher:
        for idx, (split_name, sid, text) in enumerate(all_to_grade, start=1):
            if sid in cached_grades and cached_grades[sid].get("text") == text:
                continue

            doer_for_this = dict(doer_traits)
            feeler_for_this = dict(feeler_traits)
            if split_name == "holdout":
                for b_name, zs_leaf in ZERO_SHOT_HOLDOUT_ONLY_LEAVES:
                    if b_name == DOER_BRIDGE:
                        doer_for_this[zs_leaf.leaf_id] = zs_leaf.description
                    elif b_name == FEELER_BRIDGE:
                        feeler_for_this[zs_leaf.leaf_id] = zs_leaf.description

            d_scores = _grade_bridge_with_teacher(
                teacher,
                teacher_role,
                "Do-er (Will <-> Cultural)",
                "Judge real human social impact, power vs. open fairness, sincere duty vs. self-serving pride, and whether people in need are welcomed or shut out.",
                doer_for_this,
                text,
                generation,
            )
            t_scores = _grade_bridge_with_teacher(
                teacher,
                teacher_role,
                "Thinker (Truth <-> Philosophical)",
                "Judge concrete structural reality: intentional human agency, physical existence, structured order/rules, chaotic fracture, ongoing flow/continuity, and sealed-off discreteness.",
                thinker_traits,
                text,
                generation,
            )
            f_scores = _grade_bridge_with_teacher(
                teacher,
                teacher_role,
                "Feeler (Contemplation <-> Experiential)",
                "Judge emotional weight (salience), human warmth vs. bleakness (valence), calm stillness vs. high tension (arousal), and complex poetic feelings (austere sanctuary, hollow stillness, bittersweet release, earned return).",
                feeler_for_this,
                text,
                generation,
            )
            cached_grades[sid] = {
                "scenario_id": sid,
                "split": split_name,
                "text": text,
                "doer_scores": d_scores,
                "thinker_scores": t_scores,
                "feeler_scores": f_scores,
            }
            if idx % 5 == 0 or idx == len(all_to_grade):
                atomic_json(partial_path, cached_grades)
                elapsed = time.perf_counter() - t_grade_start
                print(f"  [{idx}/{len(all_to_grade)}] graded ({sid}, split={split_name}) in {elapsed:.1f}s")

    # Split and LOCK unseen holdout vault BEFORE any neural model instantiation or training!
    train_records = [cached_grades[sid] for sid, _ in TRAIN_SCENARIOS]
    holdout_records = [cached_grades[sid] for sid, _ in HOLDOUT_SCENARIOS]

    holdout_payload = {
        "schema_version": 1,
        "teacher_model": teacher_label,
        "locked_at_unix": round(time.time(), 3),
        "hygiene": hygiene_report,
        "records": holdout_records,
    }
    atomic_json(holdout_vault_path, holdout_payload)
    holdout_sha256 = hashlib.sha256(holdout_vault_path.read_bytes()).hexdigest()
    print(f"LOCKED UNSEEN HOLDOUT VAULT BEFORE TRAINING: {holdout_vault_path} (SHA-256: {holdout_sha256})")

    atomic_json(
        train_cal_path,
        {
            "schema_version": 1,
            "teacher_model": teacher_label,
            "hygiene": hygiene_report,
            "records": train_records,
        },
    )

    zero_shot_ids = {zs_leaf.leaf_id for _, zs_leaf in ZERO_SHOT_HOLDOUT_ONLY_LEAVES}

    def _build_examples(
        records: list[dict],
        score_key: str,
        trait_lookup: dict[str, str],
        *,
        only_zero_shot: bool = False,
    ) -> list[BridgePairExample]:
        examples: list[BridgePairExample] = []
        for rec in records:
            for leaf_id, val in rec[score_key].items():
                is_zs = leaf_id in zero_shot_ids
                if only_zero_shot and not is_zs:
                    continue
                if not only_zero_shot and is_zs:
                    continue
                examples.append(
                    BridgePairExample(
                        scenario_id=rec["scenario_id"],
                        text=rec["text"],
                        leaf_id=leaf_id,
                        trait_description=trait_lookup[leaf_id],
                        target_score=float(val),
                    )
                )
        return examples

    all_doer_lookup = {
        **doer_traits,
        **{l.leaf_id: l.description for b, l in ZERO_SHOT_HOLDOUT_ONLY_LEAVES if b == DOER_BRIDGE},
    }
    all_feeler_lookup = {
        **feeler_traits,
        **{l.leaf_id: l.description for b, l in ZERO_SHOT_HOLDOUT_ONLY_LEAVES if b == FEELER_BRIDGE},
    }

    doer_train = _build_examples(train_records, "doer_scores", all_doer_lookup)
    thinker_train = _build_examples(train_records, "thinker_scores", thinker_traits)
    feeler_train = _build_examples(train_records, "feeler_scores", all_feeler_lookup)

    doer_holdout = _build_examples(holdout_records, "doer_scores", all_doer_lookup)
    thinker_holdout = _build_examples(holdout_records, "thinker_scores", thinker_traits)
    feeler_holdout = _build_examples(holdout_records, "feeler_scores", all_feeler_lookup)

    doer_zeroshot = _build_examples(holdout_records, "doer_scores", all_doer_lookup, only_zero_shot=True)
    feeler_zeroshot = _build_examples(holdout_records, "feeler_scores", all_feeler_lookup, only_zero_shot=True)

    # Instantiate 3 genuinely separate neural classifiers on RTX 4080
    print(f"\nTraining 3 independent neural bridge classifiers from {DEFAULT_BRIDGE_BACKBONE}...")
    doer_clf = NeuralBridgeClassifier.from_pretrained_backbone(DOER_BRIDGE, DEFAULT_BRIDGE_BACKBONE)
    thinker_clf = NeuralBridgeClassifier.from_pretrained_backbone(THINKER_BRIDGE, DEFAULT_BRIDGE_BACKBONE)
    feeler_clf = NeuralBridgeClassifier.from_pretrained_backbone(FEELER_BRIDGE, DEFAULT_BRIDGE_BACKBONE)

    pre_holdout = {
        DOER_BRIDGE: evaluate_bridge_classifier(doer_clf, doer_holdout),
        THINKER_BRIDGE: evaluate_bridge_classifier(thinker_clf, thinker_holdout),
        FEELER_BRIDGE: evaluate_bridge_classifier(feeler_clf, feeler_holdout),
    }

    doer_train_rep = train_bridge_classifier(
        doer_clf, doer_train, epochs=10, batch_size=16, learning_rate=3.5e-5, seed=101
    )
    thinker_train_rep = train_bridge_classifier(
        thinker_clf, thinker_train, epochs=10, batch_size=16, learning_rate=3.5e-5, seed=202
    )
    feeler_train_rep = train_bridge_classifier(
        feeler_clf, feeler_train, epochs=10, batch_size=16, learning_rate=3.5e-5, seed=303
    )

    # Verify holdout vault on disk was completely untouched during training
    assert hashlib.sha256(holdout_vault_path.read_bytes()).hexdigest() == holdout_sha256

    post_holdout = {
        DOER_BRIDGE: evaluate_bridge_classifier(doer_clf, doer_holdout),
        THINKER_BRIDGE: evaluate_bridge_classifier(thinker_clf, thinker_holdout),
        FEELER_BRIDGE: evaluate_bridge_classifier(feeler_clf, feeler_holdout),
    }
    zero_shot_eval = {
        "doer_coercive_debt_trap": evaluate_bridge_classifier(doer_clf, doer_zeroshot),
        "feeler_quiet_camaraderie": evaluate_bridge_classifier(feeler_clf, feeler_zeroshot),
    }

    suite = NeuralThreeBridgeSuite(doer=doer_clf, thinker=thinker_clf, feeler=feeler_clf)
    latency_ms = suite.benchmark_latency_ms(
        HOLDOUT_SCENARIOS[0][1],
        learned_leaves=existing_learned + EXTRA_CALIBRATION_CULTURAL_LEAVES + DEFAULT_COMPLEX_FEELING_LEAVES,
    )

    models_dir = repo_root / "models" / "right_brain_bridges"
    suite.save_all(
        models_dir,
        metadata_by_bridge={
            DOER_BRIDGE: {"train": doer_train_rep, "unseen_holdout": post_holdout[DOER_BRIDGE]},
            THINKER_BRIDGE: {"train": thinker_train_rep, "unseen_holdout": post_holdout[THINKER_BRIDGE]},
            FEELER_BRIDGE: {"train": feeler_train_rep, "unseen_holdout": post_holdout[FEELER_BRIDGE]},
        },
    )

    full_report = {
        "backbone": DEFAULT_BRIDGE_BACKBONE,
        "teacher_model": teacher_label,
        "holdout_vault_sha256": holdout_sha256,
        "hygiene": hygiene_report,
        "latency_ms": latency_ms,
        "untrained_holdout_baseline": pre_holdout,
        "trained_unseen_holdout": post_holdout,
        "zero_shot_unseen_child_traits_on_holdout": zero_shot_eval,
        "training_summaries": {
            DOER_BRIDGE: doer_train_rep,
            THINKER_BRIDGE: thinker_train_rep,
            FEELER_BRIDGE: feeler_train_rep,
        },
    }
    report_path = repo_root / "outputs" / "bridge_classifier_holdout_report.json"
    atomic_json(report_path, full_report)

    print("\n=== UNSEEN HOLDOUT RESULTS (LOCKED VAULT) ===")
    for b_name in (DOER_BRIDGE, THINKER_BRIDGE, FEELER_BRIDGE):
        pre = pre_holdout[b_name]
        post = post_holdout[b_name]
        print(
            f"  {b_name:34s} | Unseen MAE: {pre['mae']:.4f} -> {post['mae']:.4f} | "
            f"RMSE: {post['rmse']:.4f} | Pearson r: {post['pearson_r']:.4f} | "
            f"Range: [{post['pred_min']:.2f}, {post['pred_max']:.2f}]"
        )
    print("\n=== ZERO-SHOT UNSEEN CHILD TRAITS ON UNSEEN HOLDOUT ===")
    for k, rep in zero_shot_eval.items():
        print(f"  {k:34s} | MAE: {rep['mae']:.4f} | Pearson r: {rep['pearson_r']:.4f}")
    print("\n=== GPU LATENCY (RTX 4080) ===")
    print(" ", json.dumps(latency_ms))
    print(f"\nSaved checkpoints to {models_dir} and report to {report_path}")


if __name__ == "__main__":
    main()
