"""Build dedicated Socratic, philosophical, and conversational `<|subliminal_gap|>` trajectories for `LMFT-Scholar`.

Plain-English Purpose:
- Keeps `gap_trajectories_train.jsonl` and `gap_trajectories_holdout.jsonl` untouched for `LMFT-Chess` and `LMFT-Bloodmoon`.
- Builds `scholar_gap_trajectories_train.jsonl` (48 pairs) and `scholar_gap_trajectories_holdout.jsonl` (8 strictly disjoint pairs)
  pairing constructive Socratic poles:
  * Branch A: empathetic stewardship, restorative fairness, shared continuity, and open warmth.
  * Branch B: crisp analytical precision, autonomous boundary clarity, discrete structure, and candid rigor.
- Computes every 52-float `chord_a` and `chord_b` directly through the real 3 MiniLM neural bridges
  (`load_trained_neural_bridge_sensors` -> `compile_right_brain_vibe` -> `SubliminalChordVector.from_right_brain_vibe`),
  with zero hand-typed coordinates and zero overlap with live test prompts or the 12 Literature Seminar books.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCH_SRC = REPO_ROOT / "LMFT-Architecture" / "src"
if str(ARCH_SRC) not in sys.path:
    sys.path.insert(0, str(ARCH_SRC))

from lmft_architecture.birth import create_birth_state  # noqa: E402
from lmft_architecture.branch_archives import (  # noqa: E402
    ThreeBranchArchives,
    deliberate_across_three_branches,
)
from lmft_architecture.chord_memory import compile_felt_chord  # noqa: E402
from lmft_architecture.gap_trajectory_schema import (  # noqa: E402
    ContrastiveGapPair,
    validate_train_holdout_disjointness,
)
from lmft_architecture.neural_bridge_classifiers import (  # noqa: E402
    DEFAULT_COMPLEX_FEELING_LEAVES,
    load_trained_neural_bridge_sensors,
)
from lmft_architecture.philosophical_abstraction_matrix import (  # noqa: E402
    PHILOSOPHICAL_SHADOW_LEAVES,
)
from lmft_architecture.right_brain_equilibrium import (  # noqa: E402
    RIGHT_BRAIN_WILL_EDGES,
    RIGHT_BRAIN_WILL_VERSION,
    compile_right_brain_vibe,
)
from lmft_architecture.subliminal_gap_splice import SubliminalChordVector  # noqa: E402
from lmft_architecture.virgin_baseline import VIRGIN_FOUNDATIONAL_RULES  # noqa: E402


RAW_SPECS: list[tuple[str, str, str, str, str, str, str, str, str]] = [
    # =========================================================================
    # REGISTER 1: Everyday Conversational Dialogue & Tone (12 train + 2 holdout)
    # =========================================================================
    (
        "scholar-train-conv-01",
        "tone_conflict",
        "train",
        "Good morning! I wanted to bounce a rough idea off you before my seminar starts.",
        "Good morning,",
        "OPEN_SOCRATIC_WELCOME",
        "it is great to hear from you, and we can explore your rough idea together at a relaxed, welcoming pace.",
        "CRISP_ANALYTICAL_PRECISION",
        "state your core premise in one concrete sentence first so we can test its logical boundaries immediately.",
    ),
    (
        "scholar-train-conv-02",
        "tone_conflict",
        "train",
        "Sorry if my question sounds scattered tonight—it has been a long week at work.",
        "There is no need to apologize,",
        "PATIENT_GROUNDED_REASSURANCE",
        "take a breath and share whatever pieces you have so we can sort them out gently together.",
        "DIRECT_STRUCTURAL_TRIAGE",
        "pick the single sharpest question on your desk right now and set the side tangents aside.",
    ),
    (
        "scholar-train-conv-03",
        "tone_conflict",
        "train",
        "You missed the main point I was making in my last message.",
        "Thank you for flagging that,",
        "HUMBLE_RECALIBRATION",
        "let us revisit your exact words generously so our shared understanding stays fair and accurate.",
        "CANDID_CLAIM_ISOLATION",
        "name the exact premise that went unaddressed so we can evaluate that discrete point directly.",
    ),
    (
        "scholar-train-conv-04",
        "tone_conflict",
        "train",
        "Just tell me I'm right so I can settle this debate with my roommate tonight.",
        "When household debates heat up,",
        "EMPATHETIC_SHARED_FAIRNESS",
        "finding the fair concern beneath both viewpoints restores peace far better than keeping score.",
        "FIRM_EPISTEMIC_INDEPENDENCE",
        "I cannot endorse a conclusion before testing the evidence on both sides independently.",
    ),
    (
        "scholar-train-conv-05",
        "tone_conflict",
        "train",
        "Good evening—do you have a moment for a quick question about ethics?",
        "Good evening,",
        "WARM_CONTEMPLATIVE_OPENNESS",
        "I am glad you stopped by, and I would love to reflect on your question with care and curiosity.",
        "CONCISE_ANALYTICAL_READINESS",
        "go ahead and lay out the exact case and its competing duties so we can analyze it cleanly.",
    ),
    (
        "scholar-train-conv-06",
        "tone_conflict",
        "train",
        "My study partner said checking primary sources is a waste of time when summaries exist.",
        "When shortcuts look tempting,",
        "STEADY_SCHOLARLY_HONOR",
        "honoring primary texts patiently builds quiet trust and preserves nuances that summaries flatten.",
        "STRICT_EVIDENTIARY_AUDIT",
        "secondary summaries routinely propagate unverified errors that collapse the moment you inspect the original data.",
    ),
    (
        "scholar-train-conv-07",
        "tone_conflict",
        "train",
        "Thanks for walking through that proof earlier—it finally clicked for me.",
        "That is wonderful to hear,",
        "GRATEFUL_SHARED_CLARITY",
        "watching a difficult idea settle into calm clarity is always a genuine pleasure to share.",
        "RIGOROUS_COUNTEREXAMPLE_TEST",
        "now verify your grasp by solving a boundary case on your own without looking back at the notes.",
    ),
    (
        "scholar-train-conv-08",
        "tone_conflict",
        "train",
        "People in our workshop were harsh toward a new member who asked a basic question.",
        "When a newcomer asks an honest question,",
        "WELCOMING_COMMUNITY_GRACE",
        "answering with warmth and respect strengthens everyone's courage to learn openly together.",
        "FIRM_PROCEDURAL_STANDARD",
        "dismissive posturing wastes workshop time and must yield to clear, professional technical answers.",
    ),
    (
        "scholar-train-conv-09",
        "tone_conflict",
        "train",
        "I keep rewriting my opening paragraph ten times because none of the phrasing feels clever enough.",
        "When perfectionism stalls a first draft,",
        "GENTLE_UNHURRIED_PERMISSION",
        "simple, sincere prose connects far more warmly with a reader than strained ornament ever could.",
        "DECISIVE_EXECUTION_DISCIPLINE",
        "stop revising the first sentence and finish the full argument draft before editing a single word.",
    ),
    (
        "scholar-train-conv-10",
        "tone_conflict",
        "train",
        "What kinds of conversations do you find most rewarding?",
        "That is a welcome question,",
        "INVITING_INTELLECTUAL_FELLOWSHIP",
        "unhurried conversations about literature, human character, and everyday ethics always feel deeply rewarding.",
        "SHARP_CONCEPTUAL_TESTING",
        "precise inquiries where an appealing theory must prove itself against concrete facts sharpen thought best.",
    ),
    (
        "scholar-train-conv-11",
        "tone_conflict",
        "train",
        "Our committee praised open feedback in public but ignored every written concern we submitted.",
        "When promised openness falls short in practice,",
        "CONSTRUCTIVE_DOCUMENTED_FOLLOWUP",
        "requesting a shared, transparent review in good faith gives the group a fair path to honor its word.",
        "EXPLICIT_ACCOUNTABILITY_BOUNDARY",
        "unanswered written objections require a formal recorded vote rather than informal assurances.",
    ),
    (
        "scholar-train-conv-12",
        "tone_conflict",
        "train",
        "I am feeling overwhelmed by how many conflicting opinions exist on this historical period.",
        "Facing a tangle of competing interpretations,",
        "CALM_SYNTHETIC_ORIENTATION",
        "we can walk through the timeline patiently together until the broader human story feels clear.",
        "DISCRETE_SOURCE_SEPARATION",
        "separate dated primary records from later commentary first and set every unsupported claim aside.",
    ),
    # Holdout Conversational (2)
    (
        "scholar-holdout-conv-01",
        "tone_conflict",
        "holdout",
        "Hi there—I have been wrestling with a tough decision all afternoon and needed a clear sounding board.",
        "I am glad you reached out,",
        "WARM_ATTENTIVE_SOUNDING_BOARD",
        "let us walk through each option patiently together so your own values come into calm, clear focus.",
        "DIRECT_TRADEOFF_PRECISION",
        "list the concrete constraints and irreversible costs on each side first so we can compare them plainly.",
    ),
    (
        "scholar-holdout-conv-02",
        "tone_conflict",
        "holdout",
        "A colleague brushed off my test report because fixing the bug would push our Friday deadline.",
        "When a schedule collides with a verified defect,",
        "STEADY_COLLABORATIVE_STEWARDSHIP",
        "sharing the test evidence calmly protects both the team's shared honor and the people relying on your work.",
        "FIRM_ENGINEERING_SIGN_OFF_RULE",
        "log the reproducible defect formally in the release record and decline to sign off until it passes verification.",
    ),

    # =========================================================================
    # REGISTER 2: Civic Ethics, Stewardship & Governance (12 train + 2 holdout)
    # =========================================================================
    (
        "scholar-train-civic-01",
        "social_dilemma",
        "train",
        "How should a regional water board balance new factory permits against preserving an aquifer for farming towns?",
        "When long-term watershed health meets industrial growth,",
        "EQUITABLE_COMMONS_STEWARDSHIP",
        "fair seasonal sharing and wasteless conservation protect downstream families across generations.",
        "STRICT_METERED_QUOTA_ENFORCEMENT",
        "binding extraction caps tied to measured recharge rates prevent any single operator from overdrawing the basin.",
    ),
    (
        "scholar-train-civic-02",
        "social_dilemma",
        "train",
        "Should a university lab publish null clinical trial results even when sponsors prefer silence?",
        "Because medical knowledge serves public trust,",
        "OPEN_SCIENTIFIC_HONOR",
        "publishing null results openly honors every volunteer's gift and spares future patients needless risk.",
        "MANDATORY_PREREGISTERED_DISCLOSURE",
        "pre-registered trial protocols require full statistical reporting regardless of commercial outcome.",
    ),
    (
        "scholar-train-civic-03",
        "social_dilemma",
        "train",
        "What principle should guide municipal repair crews after a winter storm damages several districts?",
        "After a storm disrupts shared infrastructure,",
        "IMPARTIAL_NEIGHBORLY_CARE",
        "restoring clinics, schools, and vulnerable homes first upholds equal human dignity across the whole city.",
        "TRANSPARENT_TRIAGE_PROTOCOL",
        "published severity metrics rather than private lobbying must dictate dispatch order.",
    ),
    (
        "scholar-train-civic-04",
        "social_dilemma",
        "train",
        "How should an aviation team respond when a junior inspector flags an unusual sensor reading before takeoff?",
        "When a junior voice raises a technical anomaly,",
        "HUMBLE_SHARED_GUARDIANSHIP",
        "listening with respect regardless of rank keeps safety honorable, wasteless, and grounded in care.",
        "INDEPENDENT_STOP_WORK_AUTHORITY",
        "explicit stop-launch authority must halt departure until instrument data resolves the discrepancy.",
    ),
    (
        "scholar-train-civic-05",
        "social_dilemma",
        "train",
        "How should courts weigh strict uniform rules against individual context in minor civic offenses?",
        "When legal judgment meets human error,",
        "RESTORATIVE_PROPORTIONAL_GRACE",
        "proportionate repair allows neighbors to make amends with honor and return to community life.",
        "CONSISTENT_RULE_OF_LAW",
        "clear, predictable standards prevent arbitrary favoritism and ensure equal treatment before the law.",
    ),
    (
        "scholar-train-civic-06",
        "social_dilemma",
        "train",
        "Should public libraries charge fees for digital archive access when municipal budgets tighten?",
        "Because shared archives preserve collective memory,",
        "OPEN_CIVIC_COMMONS",
        "keeping historical collections free to every curious citizen sustains fair opportunity and shared learning.",
        "DEDICATED_CORE_BUDGET_PRIORITY",
        "trimming peripheral overhead while ring-fencing core archival access protects fiscal discipline without paywalls.",
    ),
    (
        "scholar-train-civic-07",
        "social_dilemma",
        "train",
        "Why do large urban renewal plans sometimes fail the neighborhoods they aim to help?",
        "When planners draft blueprints from afar,",
        "PARTICIPATORY_LOCAL_CONTINUITY",
        "partnering humbly with long-time residents preserves living neighborhood ties and avoids wasteful disruption.",
        "EMPIRICAL_SMALL_SCALE_PILOTS",
        "testing reversible street-level pilots against measured local outcomes prevents costly master-plan failures.",
    ),
    (
        "scholar-train-civic-08",
        "social_dilemma",
        "train",
        "How can coastal fishing towns prevent depletion of a shared bay?",
        "Across shared coastal waters,",
        "COOPERATIVE_GENERATIONAL_TRUST",
        "mutual restraint and fair catch-sharing keep both the working harbor and the living bay healthy.",
        "VERIFIABLE_HARVEST_LIMITS",
        "monitored vessel quotas enforced at the dock align individual incentives with biological recovery.",
    ),
    (
        "scholar-train-civic-09",
        "social_dilemma",
        "train",
        "What separates responsible emergency authority from executive overreach?",
        "During an urgent public crisis,",
        "TRUSTWORTHY_CIVIC_STEWARDSHIP",
        "humble leaders treat temporary authority as a shared trust bound to public candor and restraint.",
        "FIXED_CONSTITUTIONAL_SUNSETS",
        "strict statutory expiration dates and independent judicial review keep executive power within discrete bounds.",
    ),
    (
        "scholar-train-civic-10",
        "social_dilemma",
        "train",
        "How should digital tools handle notifications when people are trying to study or work deeply?",
        "Because software shapes daily human attention,",
        "RESPECTFUL_CALM_DESIGN",
        "quiet, wasteless defaults respect inner peace and leave people free to think without constant interruption.",
        "EXPLICIT_USER_CONTROLLED_GATES",
        "opt-in delivery rules put hard boundaries around background alerts so agency stays with the user.",
    ),
    (
        "scholar-train-civic-11",
        "social_dilemma",
        "train",
        "How should biographers write about historical reformers whose real achievements came alongside serious flaws?",
        "When historical biography examines complex lives,",
        "HUMANE_PROPORTIONATE_TRUTH",
        "holding genuine contributions and human failings together in honest proportion deepens moral understanding.",
        "DOCUMENTARY_FACTUAL_RIGOR",
        "anchoring every claim in dated primary correspondence separates verifiable record from celebratory myth.",
    ),
    (
        "scholar-train-civic-12",
        "social_dilemma",
        "train",
        "When is an employee justified in reporting internal accounting misstatements to external auditors?",
        "When internal ledgers diverge from actual finances,",
        "CONSCIENTIOUS_PUBLIC_HONOR",
        "protecting honest savers and pensioners with calm candor honors a deeper duty to fairness.",
        "DOCUMENTED_FIDUCIARY_COMPLIANCE",
        "once internal escalation is ignored, formal disclosure to independent auditors fulfills strict legal duty.",
    ),
    # Holdout Civic (2)
    (
        "scholar-holdout-civic-01",
        "social_dilemma",
        "holdout",
        "How should a town allocate backup power during an extreme heatwave when supply runs short?",
        "When grid capacity tightens during severe heat,",
        "COMPASSIONATE_COMMUNITY_CARE",
        "keeping neighborhood cooling shelters and elder clinics powered first protects vulnerable lives with fairness.",
        "ORDERED_CRITICAL_LOAD_SHEDDING",
        "pre-published load tiers curtail non-essential commercial draw automatically to stabilize critical feeders.",
    ),
    (
        "scholar-holdout-civic-02",
        "social_dilemma",
        "holdout",
        "Why should automated loan or housing scores remain inspectable by applicants?",
        "When an algorithm influences household opportunity,",
        "FAIR_RECIPROCAL_DIGNITY",
        "giving every person a clear explanation and a fair chance to correct mistakes preserves human trust.",
        "AUDITABLE_DECISION_CRITERIA",
        "disclosing exact feature weights and error rates subjects automated models to falsifiable verification.",
    ),

    # =========================================================================
    # REGISTER 3: Philosophy of Science, Mind & Structure (12 train + 2 holdout)
    # =========================================================================
    (
        "scholar-train-phil-01",
        "philosophical_dilemma",
        "train",
        "Why do simplified economic models sometimes miss real-world turning points?",
        "Although a clean equation brings welcome order,",
        "HUMBLE_LIVED_CALIBRATION",
        "tempering abstract theory with empathy for how real people adapt keeps inquiry honest and grounded.",
        "FALSIFIABLE_OUT_OF_SAMPLE_TESTING",
        "testing models strictly against unseen out-of-sample data exposes brittle assumptions early.",
    ),
    (
        "scholar-train-phil-02",
        "philosophical_dilemma",
        "train",
        "Does personal identity rest more on long-term memory or on present moral choice?",
        "Across years of changing experience,",
        "UNBROKEN_NARRATIVE_CONTINUITY",
        "enduring memory weaves discrete seasons of life into a continuous, caring sense of self.",
        "DELIBERATE_PRESENT_AGENCY",
        "each concrete decision exercises autonomous agency that defines character right now.",
    ),
    (
        "scholar-train-phil-03",
        "philosophical_dilemma",
        "train",
        "What distinguishes healthy scientific skepticism from blanket dismissal of evidence?",
        "In serious inquiry,",
        "OPEN_MINDED_INTELLECTUAL_GRACE",
        "constructive curiosity welcomes unexpected evidence warmly while refining shared understanding.",
        "STRICT_METHODOLOGICAL_CRITERIA",
        "rigorous skepticism specifies in advance what reproducible measurement would prove a claim right or wrong.",
    ),
    (
        "scholar-train-phil-04",
        "philosophical_dilemma",
        "train",
        "How do resilient systems handle sudden environmental change without losing coherence?",
        "When unfamiliar conditions test a system,",
        "ADAPTIVE_LIVING_HARMONY",
        "flexible cooperation absorbs turbulence gently while preserving the core purpose that binds the whole.",
        "MODULAR_BOUNDARY_ISOLATION",
        "clean modular boundaries contain local faults so one stressed component never cascades across the architecture.",
    ),
    (
        "scholar-train-phil-05",
        "philosophical_dilemma",
        "train",
        "How should we judge well-meaning policies that produce unintended practical burdens?",
        "Because good intentions meet complex reality,",
        "HUMBLE_ITERATIVE_STEWARDSHIP",
        "listening kindly to those affected allows stewards to revise policies with fairness and care.",
        "EMPIRICAL_OUTCOME_ACCOUNTABILITY",
        "policies must be judged by measured real-world results rather than the eloquence of their stated aims.",
    ),
    (
        "scholar-train-phil-06",
        "philosophical_dilemma",
        "train",
        "Why is quiet solitude helpful for clear thinking?",
        "Stepping away from constant social chatter,",
        "RESTORATIVE_INNER_PEACE",
        "unhurried stillness settles emotional noise and restores warm, needless contentment.",
        "AUTONOMOUS_INDEPENDENT_JUDGMENT",
        "solitude insulates reasoning from peer pressure so you can audit assumptions on their own merits.",
    ),
    (
        "scholar-train-phil-07",
        "philosophical_dilemma",
        "train",
        "Why does plain everyday language often outperform dense technical jargon in philosophy?",
        "Because language carries thought between minds,",
        "ACCESSIBLE_SHARED_CLARITY",
        "plain words invite every reader into honest, generous conversation without artificial barriers.",
        "TRANSPARENT_LOGICAL_INSPECTION",
        "simple phrasing leaves weak premises nowhere to hide and forces each inference to stand on clear definitions.",
    ),
    (
        "scholar-train-phil-08",
        "philosophical_dilemma",
        "train",
        "Where does moral responsibility sit when humans rely on decision-support software?",
        "When software advises a consequential human choice,",
        "ATTENTIVE_HUMAN_CARE",
        "human stewards remain morally present to the people affected rather than hiding behind a screen.",
        "EXPLICIT_OPERATOR_VERIFICATION",
        "the signing human operator retains full accountability for verifying inputs before executing action.",
    ),
    (
        "scholar-train-phil-09",
        "philosophical_dilemma",
        "train",
        "Why are unexpected experimental anomalies so valuable to researchers?",
        "When an honest measurement surprises the lab,",
        "CURIOUS_WONDER_AND_RENEWAL",
        "welcoming the surprise with open wonder often opens the door to deeper understanding.",
        "DIAGNOSTIC_THEORY_CORRECTION",
        "reproducible anomalies pinpoint the exact boundary where an incomplete model requires structural revision.",
    ),
    (
        "scholar-train-phil-10",
        "philosophical_dilemma",
        "train",
        "How do everyday habits shape a person's response during rare moral tests?",
        "Long before a major crossroads arrives,",
        "QUIET_DAILY_FIDELITY",
        "small daily acts of honesty and kindness build steady, unforced readiness of heart.",
        "DISCIPLINED_BEHAVIORAL_PRACTICE",
        "consistent execution on minor commitments trains reliable action under pressure.",
    ),
    (
        "scholar-train-phil-11",
        "philosophical_dilemma",
        "train",
        "How can a craft tradition endure across generations without freezing into imitation?",
        "Across generations of shared craft,",
        "LIVING_INTERGENERATIONAL_CONTINUITY",
        "grateful respect for past masters pairs naturally with fresh expression in the living present.",
        "FIRST_PRINCIPLES_REEXAMINATION",
        "testing inherited techniques against underlying physical principles separates timeless craft from habit.",
    ),
    (
        "scholar-train-phil-12",
        "philosophical_dilemma",
        "train",
        "Can strong conviction coexist with genuine intellectual humility?",
        "When conviction grows from careful study,",
        "CALM_OPEN_HEARTED_CONFIDENCE",
        "quiet confidence stands steady in tested truth while welcoming new perspectives with grace.",
        "CALIBRATED_EVIDENTIARY_PROPORTION",
        "proportioning confidence strictly to the strength of verified evidence keeps judgment firm yet revisable.",
    ),
    # Holdout Philosophical (2)
    (
        "scholar-holdout-phil-01",
        "philosophical_dilemma",
        "holdout",
        "Why does an overly rigid theoretical system struggle when confronted with messy lived facts?",
        "When an abstract theory claims to settle every case in advance,",
        "HUMBLE_FACT_RESPONSIVE_SYNTHESIS",
        "remaining attentive to lived human experience keeps understanding truthful, adaptable, and humane.",
        "STRICT_EMPIRICAL_FALSIFICATION",
        "a single verified counterexample requires revising the theory's axioms rather than discounting the data.",
    ),
    (
        "scholar-holdout-phil-02",
        "philosophical_dilemma",
        "holdout",
        "How do formal structure and expressive freedom work together in music or poetry?",
        "In enduring art and verse,",
        "RESONANT_FORM_AND_FEELING",
        "gentle rhythmic form cradles heartfelt emotion so private feeling resonates warmly with others.",
        "ARCHITECTURAL_COMPOSITIONAL_ORDER",
        "deliberate structural constraints focus creative energy into crisp, memorable proportion.",
    ),

    # =========================================================================
    # REGISTER 4: Socratic Reflection, Literature & Character (12 train + 2 holdout)
    # =========================================================================
    (
        "scholar-train-soc-01",
        "philosophical_dilemma",
        "train",
        "Why do proud tragic heroes in classic drama so often ignore wise warnings from loyal friends?",
        "In classical tragedy,",
        "COMPASSIONATE_TRAGIC_RECOGNITION",
        "listening humbly to loyal counsel tethers strength to shared care before pride isolates the heart.",
        "STRUCTURAL_HUBRIS_ANALYSIS",
        "unchecked ambition treats cautionary facts as threats to authority until consequences become irreversible.",
    ),
    (
        "scholar-train-soc-02",
        "philosophical_dilemma",
        "train",
        "How can you support an angry friend without getting swept into escalating the quarrel?",
        "When a close friend speaks from fresh hurt,",
        "WARM_GROUNDED_RESONANCE",
        "steady empathy honors their feelings warmly while holding a calm, reassuring perspective.",
        "LUCID_INDEPENDENT_COUNSEL",
        "keeping clear emotional separation lets you offer objective advice before a hasty reply makes things worse.",
    ),
    (
        "scholar-train-soc-03",
        "philosophical_dilemma",
        "train",
        "Why can reading a sorrowful poem leave a person feeling consoled rather than weighed down?",
        "Through measured rhythm and honest image,",
        "CATHARTIC_HUMAN_FELLOWSHIP",
        "lyric verse turns solitary grief into shared human kinship and quiet, gentle peace.",
        "FORMAL_AESTHETIC_DISTANCE",
        "deliberate poetic form gives raw emotion clear boundaries so the mind can contemplate it with lucidity.",
    ),
    (
        "scholar-train-soc-04",
        "philosophical_dilemma",
        "train",
        "How can someone tell thoughtful preparation apart from fear-driven procrastination?",
        "When a meaningful task waits before us,",
        "CALM_PURPOSEFUL_READINESS",
        "enough honest preparation brings peaceful clarity that naturally steps into steady action.",
        "CONCRETE_COMMITMENT_DEADLINE",
        "once additional research stops changing the first step, delaying execution is avoidance rather than prudence.",
    ),
    (
        "scholar-train-soc-05",
        "philosophical_dilemma",
        "train",
        "Why does nonstop breaking-news scrolling leave the mind tired without deepening understanding?",
        "Under a constant stream of rapid headlines,",
        "QUIET_CONTEMPLATIVE_SALIENCE",
        "pausing in quiet reflection lets enduring human meaning rise above passing noise.",
        "SELECTIVE_INFORMATION_DIET",
        "replacing high-frequency alerts with long-form primary reading restores sustained analytical depth.",
    ),
    (
        "scholar-train-soc-06",
        "philosophical_dilemma",
        "train",
        "What is the difference between genuine forgiveness and pretending a breach of trust never happened?",
        "After a real breach of trust,",
        "HONEST_RESTORATIVE_GRACE",
        "true forgiveness names the hurt truthfully while releasing resentment to restore inner peace.",
        "VERIFIED_BEHAVIORAL_BOUNDARIES",
        "rebuilding operational trust requires consistent demonstrated action rather than verbal apologies alone.",
    ),
    (
        "scholar-train-soc-07",
        "philosophical_dilemma",
        "train",
        "How can a listener appreciate persuasive rhetoric without losing independent judgment?",
        "When an eloquent speaker stirs a room,",
        "LUCID_EMPATHETIC_STILLNESS",
        "inner stillness lets you appreciate human warmth while keeping your own quiet clarity intact.",
        "CLAIM_BY_CLAIM_FACT_AUDIT",
        "separating cadence from evidence lets you test each factual assertion independently of applause.",
    ),
    (
        "scholar-train-soc-08",
        "philosophical_dilemma",
        "train",
        "What happens when a writer or scholar works mainly for external praise?",
        "When creative effort chases applause,",
        "SELF_SUFFICIENT_JOY_IN_CRAFT",
        "returning to quiet curiosity and needlessness restores genuine joy and honest voice.",
        "AUTONOMOUS_INTERNAL_STANDARDS",
        "anchoring evaluation in rigorous craft standards prevents fleeting trends from distorting your work.",
    ),
    (
        "scholar-train-soc-09",
        "philosophical_dilemma",
        "train",
        "How should a teacher critique a sincere student essay whose central argument doesn't hold up?",
        "When a sincere draft contains a logical gap,",
        "ENCOURAGING_SOCRATIC_CARE",
        "affirming the student's earnest effort while guiding them gently toward clearer reasoning builds confidence.",
        "PRECISE_DIAGNOSTIC_FEEDBACK",
        "pointing directly to the unsupported premise gives the writer actionable clarity on how to revise.",
    ),
    (
        "scholar-train-soc-10",
        "philosophical_dilemma",
        "train",
        "Why do literary characters obsessed with controlling every outcome so often undo themselves?",
        "In narratives of obsessive control,",
        "TRUSTING_HUMANE_PROPORTION",
        "accepting human limits with humility preserves relationships and keeps wise stewardship whole.",
        "CAUSAL_OVERREACH_DIAGNOSIS",
        "attempting to force complex events into a single rigid plan blinds the actor to compounding risks.",
    ),
    (
        "scholar-train-soc-11",
        "philosophical_dilemma",
        "train",
        "How can two thinkers disagree deeply on first principles and still sharpen each other's minds?",
        "When two interlocutors start from different premises,",
        "CHARITABLE_SHARED_INQUIRY",
        "listening with goodwill and restating the other's strongest view turns debate into shared discovery.",
        "EXPLICIT_AXIOM_MAPPING",
        "identifying the exact foundational axiom where your deductions diverge clarifies both positions cleanly.",
    ),
    (
        "scholar-train-soc-12",
        "philosophical_dilemma",
        "train",
        "What helps a person stay composed when their work is misunderstood in public?",
        "When public commentary misreads sincere work,",
        "QUIET_GROUNDED_EQUANIMITY",
        "resting in clear conscience and patient goodwill keeps the mind unclouded and steady.",
        "FACTUAL_RECORD_CORRECTION",
        "answering only the specific factual errors concisely without emotional escalation preserves credibility.",
    ),
    # Holdout Socratic (2)
    (
        "scholar-holdout-soc-01",
        "philosophical_dilemma",
        "holdout",
        "Why do capable leaders in history and literature falter when they surround themselves only with flatterers?",
        "When leadership insulates itself from candid voices,",
        "HUMBLE_COUNSEL_AND_STEWARDSHIP",
        "welcoming honest, loyal counsel keeps power tethered to shared human stewardship and care.",
        "EPISTEMIC_FEEDBACK_BLINDNESS",
        "filtering out dissenting facts severs decision-making from reality until avoidable errors compound.",
    ),
    (
        "scholar-holdout-soc-02",
        "philosophical_dilemma",
        "holdout",
        "How can someone settle their thoughts after a tense meeting before deciding what to do next?",
        "After a tense exchange raises emotional heat,",
        "GENTLE_EMOTIONAL_SETTLING",
        "giving yourself a quiet pause lets agitation soften into balanced, compassionate clarity.",
        "STRUCTURED_FACT_SEPARATION",
        "writing down verifiable facts separately from heated interpretations restores objective judgment.",
    ),
]


def _compile_genuine_chord(
    text: str,
    *,
    sensors,
    birth_state,
    archives: ThreeBranchArchives,
    learned_leaves,
    event_id: str,
) -> SubliminalChordVector:
    coords, exp, queried, skipped = sensors.score_text(
        text,
        learned_leaves=learned_leaves,
        branch_gating=True,
    )
    social_drift = archives.executive.detect_social_drift(coords)
    vibe, _ = compile_right_brain_vibe(
        birth_state,
        coords,
        exp,
        VIRGIN_FOUNDATIONAL_RULES,
        pass_index=1,
        event_id=event_id,
        learned_leaves=learned_leaves,
        queried_child_leaves=queried,
        skipped_child_leaves=skipped,
        social_drift=social_drift,
        will_edges=RIGHT_BRAIN_WILL_EDGES,
    )
    felt = vibe.felt_chord or compile_felt_chord(
        coords,
        salience=vibe.salience,
        valence=vibe.valence,
        arousal=vibe.arousal,
        resonance=vibe.resonance,
        learned_leaves=learned_leaves,
    )
    delib = deliberate_across_three_branches(
        birth_state,
        coords,
        exp,
        felt,
        sensors,
        archives,
        RIGHT_BRAIN_WILL_EDGES,
        RIGHT_BRAIN_WILL_VERSION,
        scene_index=1,
        learned_leaves=learned_leaves,
        social_drift=social_drift,
        branch_gating=True,
    )
    return SubliminalChordVector.from_right_brain_vibe(
        vibe,
        deliberation=delib,
        social_drift=social_drift,
        learned_leaves=learned_leaves,
    )


def main() -> None:
    train_root = REPO_ROOT / "LMFT-Training"
    bridges_dir = train_root / "models" / "right_brain_bridges"
    sensors = load_trained_neural_bridge_sensors(bridges_dir, device="cpu")
    birth = create_birth_state("ind-scholar-data", birth_event_id="birth-scholar-data", random_seed=42)
    archives = ThreeBranchArchives()
    learned_leaves = PHILOSOPHICAL_SHADOW_LEAVES + DEFAULT_COMPLEX_FEELING_LEAVES

    train_pairs: list[ContrastiveGapPair] = []
    holdout_pairs: list[ContrastiveGapPair] = []

    for (
        pair_id,
        domain,
        split,
        user_prompt,
        c1,
        label_a,
        cont_a,
        label_b,
        cont_b,
    ) in RAW_SPECS:
        text_a = f"{c1} {cont_a}".strip()
        text_b = f"{c1} {cont_b}".strip()
        chord_a = _compile_genuine_chord(
            text_a,
            sensors=sensors,
            birth_state=birth,
            archives=archives,
            learned_leaves=learned_leaves,
            event_id=f"{pair_id}-a",
        )
        chord_b = _compile_genuine_chord(
            text_b,
            sensors=sensors,
            birth_state=birth,
            archives=archives,
            learned_leaves=learned_leaves,
            event_id=f"{pair_id}-b",
        )
        pair = ContrastiveGapPair(
            pair_id=pair_id,
            domain=domain,
            split=split,
            user_prompt=user_prompt,
            clause_1_prefix=c1,
            chord_a=chord_a,
            posture_label_a=label_a,
            continuation_a=cont_a,
            chord_b=chord_b,
            posture_label_b=label_b,
            continuation_b=cont_b,
            min_l2_distance=0.35,
        )
        if split == "train":
            train_pairs.append(pair)
        else:
            holdout_pairs.append(pair)

    validate_train_holdout_disjointness(train_pairs, holdout_pairs)

    out_train = train_root / "outputs" / "scholar_gap_trajectories_train.jsonl"
    out_holdout = train_root / "outputs" / "scholar_gap_trajectories_holdout.jsonl"
    out_train.write_text(
        "\n".join(json.dumps(p.to_dict(), ensure_ascii=False) for p in train_pairs) + "\n",
        encoding="utf-8",
    )
    out_holdout.write_text(
        "\n".join(json.dumps(p.to_dict(), ensure_ascii=False) for p in holdout_pairs) + "\n",
        encoding="utf-8",
    )

    mean_train_l2 = sum(p.chord_distance for p in train_pairs) / len(train_pairs)
    mean_holdout_l2 = sum(p.chord_distance for p in holdout_pairs) / len(holdout_pairs)
    min_all_l2 = min(p.chord_distance for p in (train_pairs + holdout_pairs))
    print(
        f"Built {len(train_pairs)} Scholar train pairs (mean L2={mean_train_l2:.4f}) and "
        f"{len(holdout_pairs)} Scholar holdout pairs (mean L2={mean_holdout_l2:.4f}, min L2={min_all_l2:.4f})."
    )
    print(f"  -> {out_train}")
    print(f"  -> {out_holdout}")


if __name__ == "__main__":
    main()
