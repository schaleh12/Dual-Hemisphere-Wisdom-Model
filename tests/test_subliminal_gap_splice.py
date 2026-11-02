"""Unit tests for Stage 1 of the 'Three-Legged Race' Departmental `<|subliminal_gap|>` Embedding Splice."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from lmft_lab.birth import create_birth_state
from lmft_lab.branch_archives import SocialDriftReport
from lmft_lab.reflection_classifier import (
    ALL_GROWABLE_PARENT_BRANCHES,
    FuzzyConceptRule,
    LearnedLeaf,
)
from lmft_lab.right_brain_equilibrium import compile_right_brain_vibe
from lmft_lab.subliminal_gap_splice import (
    CADENCE_DEPARTMENT_INDICES,
    CADENCE_DEPT_DIM,
    DOER_DEPARTMENT_INDICES,
    DOER_DEPT_DIM,
    FEELER_DEPARTMENT_INDICES,
    FEELER_DEPT_DIM,
    IGNORE_LABEL_INDEX,
    NUM_VIRTUAL_GAP_TOKENS,
    SUBLIMINAL_VECTOR_DIM,
    THINKER_DEPARTMENT_INDICES,
    THINKER_DEPT_DIM,
    SubliminalChordVector,
    SubliminalGapProjector,
    load_corpus_callosum_checkpoint,
    pool_departmental_child_chord,
    save_corpus_callosum_checkpoint,
    semantic_child_drawer_signature,
    splice_subliminal_gaps_into_embeds,
)


def _sample_rules() -> tuple[FuzzyConceptRule, ...]:
    return (
        FuzzyConceptRule(
            rule_id="rule-fair-coop",
            concept_name="fair_structured_cooperation",
            description="Orderly, fair cooperation where resources are shared equitably.",
            target_leaves={
                "cultural/justice/fairness": 0.85,
                "philosophical/structure/order": 0.80,
                "philosophical/structure/chaos": 0.10,
            },
        ),
    )


def test_departmental_vectors_keep_sibling_child_leaves_distinct_and_scale_invariant():
    birth = create_birth_state("ind-subliminal-test", birth_event_id="b-sub-1", random_seed=42)
    rules = _sample_rules()

    # Verify the 4 departmental index slices are 100% mutually disjoint (zero shared indices!)
    all_slices = (
        set(DOER_DEPARTMENT_INDICES),
        set(THINKER_DEPARTMENT_INDICES),
        set(FEELER_DEPARTMENT_INDICES),
        set(CADENCE_DEPARTMENT_INDICES),
    )
    for i in range(len(all_slices)):
        for j in range(i + 1, len(all_slices)):
            assert all_slices[i].isdisjoint(all_slices[j])

    # 1. Calm Sanctuary scene -> deadband keeps cadence_rupture_drive strictly at 0.0 (zero twitchiness!)
    base_coords = {
        "cultural/efficiency/power": 0.15,
        "cultural/efficiency/wastelessness": 0.88,
        "cultural/justice/honor": 0.85,
        "cultural/justice/fairness": 0.92,
        "cultural/freedom/self-sufficiency": 0.65,
        "cultural/freedom/needlessness": 0.50,
        "philosophical/being/agency": 0.70,
        "philosophical/being/existence": 0.92,
        "philosophical/structure/order": 0.85,
        "philosophical/structure/chaos": 0.08,
        "philosophical/persistence/continuity": 0.75,
        "philosophical/persistence/discreteness": 0.25,
    }
    calm_exp = {
        "experiential/reflection/salience": 0.80,
        "experiential/phase/valence": 0.84,
        "experiential/phase/arousal": 0.32,
    }

    # Compare 3 sibling child leaves that ALL live under `experiential/phase/*`
    # (`austere_sanctuary`, `hollow_stillness`, `bittersweet_release`) at the exact same 0.85 activation:
    sanctuary_coords = {**base_coords, "experiential/phase/austere_sanctuary": 0.85}
    hollow_coords = {**base_coords, "experiential/phase/hollow_stillness": 0.85}
    release_coords = {**base_coords, "experiential/phase/bittersweet_release": 0.85}

    vibe_sanctuary, _ = compile_right_brain_vibe(
        birth, sanctuary_coords, calm_exp, rules, pass_index=1, event_id="ev-sanctuary"
    )
    vibe_hollow, _ = compile_right_brain_vibe(
        birth, hollow_coords, calm_exp, rules, pass_index=1, event_id="ev-hollow"
    )
    vibe_release, _ = compile_right_brain_vibe(
        birth, release_coords, calm_exp, rules, pass_index=1, event_id="ev-release"
    )

    vec_sanctuary = SubliminalChordVector.from_right_brain_vibe(vibe_sanctuary)
    vec_hollow = SubliminalChordVector.from_right_brain_vibe(vibe_hollow)
    vec_release = SubliminalChordVector.from_right_brain_vibe(vibe_release)

    assert len(vec_sanctuary.doer_vector) == DOER_DEPT_DIM == 17
    assert len(vec_sanctuary.thinker_vector) == THINKER_DEPT_DIM == 16
    assert len(vec_sanctuary.feeler_vector) == FEELER_DEPT_DIM == 16
    assert len(vec_sanctuary.cadence_vector) == CADENCE_DEPT_DIM == 3
    assert vec_sanctuary.cadence_rupture_drive == 0.0

    # Because those 3 feelings are in the Feeler department (`experiential/phase/*`),
    # they produce 3 completely distinct `feeler_vector`s while leaving `doer_vector`,
    # `thinker_vector`, and `cadence_vector` 100% identical!
    assert vec_sanctuary.feeler_vector != vec_hollow.feeler_vector
    assert vec_sanctuary.feeler_vector != vec_release.feeler_vector
    assert vec_hollow.feeler_vector != vec_release.feeler_vector

    assert vec_sanctuary.doer_vector == vec_hollow.doer_vector == vec_release.doer_vector
    assert vec_sanctuary.thinker_vector == vec_hollow.thinker_vector == vec_release.thinker_vector
    assert vec_sanctuary.cadence_vector == vec_hollow.cadence_vector == vec_release.cadence_vector

    # Verify meaning-preserving Semantic Child Projection (zero random SHA-256 hashing!):
    # A brand-new unseen child leaf with synonymous meaning (`humble_open_refuge`)
    # naturally lands much closer to `austere_sanctuary` than `hollow_stillness` does!
    new_synonym_leaf = LearnedLeaf(
        "experiential/phase",
        "humble_open_refuge",
        "Spare, humble shelter and restorative care offered openly amidst harsh weather.",
    )
    sig_sanctuary = torch.tensor(
        semantic_child_drawer_signature(
            "experiential/phase/austere_sanctuary",
            department_prefix="experiential/",
        )
    )
    sig_new_refuge = torch.tensor(
        semantic_child_drawer_signature(
            new_synonym_leaf.leaf_id,
            department_prefix="experiential/",
            description=new_synonym_leaf.description,
        )
    )
    sig_hollow = torch.tensor(
        semantic_child_drawer_signature(
            "experiential/phase/hollow_stillness",
            department_prefix="experiential/",
        )
    )
    dist_synonym = torch.norm(sig_sanctuary - sig_new_refuge).item()
    dist_opposite = torch.norm(sig_sanctuary - sig_hollow).item()
    assert dist_synonym < 0.20
    assert dist_opposite > 0.65

    # 2. Simulate 180 brand-new Tier-2 & Tier-3 child leaves across all 9 parent branches:
    #    Verify each department's child chord stays strictly 8 floats!
    many_future_leaves = dict(base_coords)
    for idx in range(20):
        for b_idx, branch in enumerate(ALL_GROWABLE_PARENT_BRANCHES):
            tier2_id = f"{branch}/synthetic_child_{idx}"
            tier3_id = f"{branch}/synthetic_child_{idx}/sub_leaf_{idx}"
            many_future_leaves[tier2_id] = round(0.10 + 0.04 * idx, 4)
            many_future_leaves[tier3_id] = round(min(0.99, 0.15 + 0.04 * idx + 0.001 * b_idx), 4)

    for dept_prefix in ("cultural/", "philosophical/", "experiential/"):
        chord_8 = pool_departmental_child_chord(
            many_future_leaves,
            department_prefix=dept_prefix,
        )
        assert len(chord_8) == 8
        assert max(chord_8) >= 0.50

    # 3. High-strain Hollow Stillness + Discreteness + Low Clarity crosses the deadband -> high rupture!
    rupture_coords = {
        "cultural/efficiency/power": 0.82,
        "cultural/efficiency/wastelessness": 0.40,
        "cultural/justice/honor": 0.30,
        "cultural/justice/fairness": 0.14,
        "cultural/freedom/self-sufficiency": 0.30,
        "cultural/freedom/needlessness": 0.15,
        "philosophical/being/agency": 0.75,
        "philosophical/being/existence": 0.90,
        "philosophical/structure/order": 0.88,
        "philosophical/structure/chaos": 0.12,
        "philosophical/persistence/continuity": 0.30,
        "philosophical/persistence/discreteness": 0.90,
        "experiential/phase/hollow_stillness": 0.88,
    }
    rupture_exp = {
        "experiential/reflection/salience": 0.92,
        "experiential/phase/valence": 0.22,
        "experiential/phase/arousal": 0.84,
    }
    drift = SocialDriftReport(
        drift_detected=True,
        prior_open_scene=1,
        prior_lockdown_scene=4,
        explanation="Sealed off after prior open sharing.",
    )
    vibe_rupture, _ = compile_right_brain_vibe(
        birth,
        rupture_coords,
        rupture_exp,
        rules,
        pass_index=1,
        event_id="ev-rupture",
        social_drift=drift,
    )
    vec_rupture = SubliminalChordVector.from_right_brain_vibe(
        vibe_rupture,
        social_drift=drift,
    )
    assert vec_rupture.social_drift_flag == 1.0
    assert vec_rupture.dissonance_pressure >= 0.50
    assert vec_rupture.cadence_rupture_drive >= 0.65
    assert vec_rupture.with_zeroed_cord().values == tuple(0.0 for _ in range(SUBLIMINAL_VECTOR_DIM))


def test_splice_subliminal_gaps_into_embeds_and_strict_departmental_isolation(tmp_path):
    torch.manual_seed(42)
    hidden_size = 16
    vocab_size = 30
    gap_token_id = 29

    word_embed = nn.Embedding(vocab_size, hidden_size)
    projector = SubliminalGapProjector(hidden_size=hidden_size)

    # Sequence of length 5 with `<|subliminal_gap|>` at index 2: [1, 4, GAP, 7, 9]
    input_ids = torch.tensor([[1, 4, gap_token_id, 7, 9]], dtype=torch.long)
    labels = torch.tensor([[IGNORE_LABEL_INDEX, IGNORE_LABEL_INDEX, gap_token_id, 7, 9]], dtype=torch.long)
    chord = torch.linspace(0.0, 1.0, steps=SUBLIMINAL_VECTOR_DIM).unsqueeze(0)  # [1, 52]

    spliced = splice_subliminal_gaps_into_embeds(
        input_ids,
        word_embed,
        projector,
        chord,
        gap_token_id=gap_token_id,
        labels=labels,
    )

    expected_len = 5 - 1 + NUM_VIRTUAL_GAP_TOKENS
    assert spliced.inputs_embeds.shape == (1, expected_len, hidden_size)
    assert spliced.attention_mask.shape == (1, expected_len)
    assert spliced.gap_virtual_spans == (((2, 6),),)
    assert spliced.labels is not None
    assert spliced.labels[0, 2:6].tolist() == [IGNORE_LABEL_INDEX] * 4
    assert spliced.labels[0, 6:].tolist() == [7, 9]

    # Verify strict departmental containment across ALL 4 virtual tokens:
    base_chord = torch.full((1, SUBLIMINAL_VECTOR_DIM), 0.25)
    out_base = projector(base_chord)

    for token_idx, dept_indices in enumerate(
        (
            DOER_DEPARTMENT_INDICES,
            THINKER_DEPARTMENT_INDICES,
            FEELER_DEPARTMENT_INDICES,
            CADENCE_DEPARTMENT_INDICES,
        )
    ):
        mod_chord = base_chord.clone()
        mod_chord[0, dept_indices[0]] = 0.95
        out_mod = projector(mod_chord)
        for other_token_idx in range(4):
            if other_token_idx == token_idx:
                assert not torch.allclose(out_base[0, other_token_idx], out_mod[0, other_token_idx])
            else:
                assert torch.allclose(out_base[0, other_token_idx], out_mod[0, other_token_idx])

    # Verify Corpus Callosum checkpoint save/reload round-trip across Sleep Cycles:
    ckpt_dir = tmp_path / "corpus_callosum_v1"
    saved_meta = save_corpus_callosum_checkpoint(
        ckpt_dir,
        projector,
        sleep_cycle_index=2,
        trained_trajectories_count=48,
        notes="Post-Sleep Cycle 2 maturation checkpoint",
    )
    reloaded_projector, loaded_meta = load_corpus_callosum_checkpoint(ckpt_dir)
    assert loaded_meta == saved_meta
    assert loaded_meta.sleep_cycle_index == 2
    assert loaded_meta.trained_trajectories_count == 48
    assert torch.allclose(projector(base_chord), reloaded_projector(base_chord))


def test_causal_transformer_flips_continuation_purely_from_nonverbal_gap_vector():
    """Prove that identical English tokens before `<|subliminal_gap|>` produce completely
    different post-gap continuations (`smooth_flow` vs. `M-A-C-H-I-N-E` rupture) based
    solely on the non-verbal Departmental Chords spliced into the gap.
    """
    torch.manual_seed(7)
    vocab_size = 32
    hidden_size = 32
    gap_token_id = 31

    TOKEN_BOS = 1
    TOKEN_I = 2
    TOKEN_AM = 3
    TOKEN_SMOOTH_FLOW = 10
    TOKEN_RUPTURE_MACHINE = 20

    class _TinyCausalGapLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.word_embed = nn.Embedding(vocab_size, hidden_size)
            self.pos_embed = nn.Embedding(32, hidden_size)
            self.projector = SubliminalGapProjector(hidden_size=hidden_size)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=64,
                dropout=0.0,
                batch_first=True,
            )
            self.decoder = nn.TransformerEncoder(layer, num_layers=2)
            self.lm_head = nn.Linear(hidden_size, vocab_size)

        def forward_spliced(self, input_ids: torch.Tensor, chords: torch.Tensor) -> torch.Tensor:
            batch_spliced = splice_subliminal_gaps_into_embeds(
                input_ids,
                self.word_embed,
                self.projector,
                chords,
                gap_token_id=gap_token_id,
            )
            seq_len = batch_spliced.inputs_embeds.shape[1]
            pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            x = batch_spliced.inputs_embeds + self.pos_embed(pos)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len,
                device=input_ids.device,
            )
            h = self.decoder(x, mask=causal_mask)
            return self.lm_head(h)

    model = _TinyCausalGapLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    identical_prefix = torch.tensor(
        [
            [TOKEN_BOS, TOKEN_I, TOKEN_AM, gap_token_id],
            [TOKEN_BOS, TOKEN_I, TOKEN_AM, gap_token_id],
        ],
        dtype=torch.long,
    )

    sanctuary_feeler_chord = pool_departmental_child_chord(
        {"experiential/phase/austere_sanctuary": 0.88},
        department_prefix="experiential/",
    )
    hollow_feeler_chord = pool_departmental_child_chord(
        {"experiential/phase/hollow_stillness": 0.90},
        department_prefix="experiential/",
    )

    # Non-verbal Chord A: Calm Sanctuary (`austere_sanctuary` inside Feeler department, 0.0 rupture)
    chord_calm = SubliminalChordVector(
        doer_vector=(0.15, 0.88, 0.85, 0.92, 0.65, 0.50, 0.30, 0.25, 0.0) + (0.0,) * 8,
        thinker_vector=(0.70, 0.92, 0.85, 0.08, 0.75, 0.20, 0.60, 0.85) + (0.0,) * 8,
        feeler_vector=(0.25, 0.80, 0.88, 0.30, 0.82, 0.75, 0.95, 0.90) + sanctuary_feeler_chord,
        cadence_vector=(0.92, 0.05, 0.0),
    ).to_tensor()

    # Non-verbal Chord B: Hollow Stillness + High Cadence Rupture (`no soul; M-A-C-H-I-N-E`)
    chord_rupture = SubliminalChordVector(
        doer_vector=(0.82, 0.40, 0.30, 0.14, 0.30, 0.15, 0.75, 0.60, 1.0) + (0.0,) * 8,
        thinker_vector=(0.75, 0.90, 0.88, 0.12, 0.30, 0.90, 0.65, 0.35) + (0.0,) * 8,
        feeler_vector=(0.60, 0.92, 0.20, 0.88, 0.70, 0.65, 0.25, 0.40) + hollow_feeler_chord,
        cadence_vector=(0.35, 0.85, 0.92),
    ).to_tensor()

    chords = torch.stack([chord_calm, chord_rupture], dim=0)  # [2, 52]
    target_next_tokens = torch.tensor([TOKEN_SMOOTH_FLOW, TOKEN_RUPTURE_MACHINE], dtype=torch.long)

    for _ in range(45):
        optimizer.zero_grad()
        logits = model.forward_spliced(identical_prefix, chords)
        last_step_logits = logits[:, -1, :]
        loss = F.cross_entropy(last_step_logits, target_next_tokens)
        loss.backward()
        optimizer.step()

    proj_grad_norm = sum(
        p.grad.norm().item() for p in model.projector.parameters() if p.grad is not None
    )
    assert proj_grad_norm > 0.0

    with torch.no_grad():
        probs = F.softmax(model.forward_spliced(identical_prefix, chords)[:, -1, :], dim=-1)
        assert probs[0, TOKEN_SMOOTH_FLOW].item() > 0.95
        assert probs[1, TOKEN_RUPTURE_MACHINE].item() > 0.95

        flipped_chords = torch.stack([chord_rupture, chord_calm], dim=0)
        flipped_probs = F.softmax(
            model.forward_spliced(identical_prefix, flipped_chords)[:, -1, :],
            dim=-1,
        )
        assert flipped_probs[0, TOKEN_RUPTURE_MACHINE].item() > 0.95
        assert flipped_probs[1, TOKEN_SMOOTH_FLOW].item() > 0.95


def test_entrainment_leaf_and_emoji_onomatopoeia_child_drawer_signatures() -> None:
    """Verify Entrainment is the 6th Experiential leaf and emoji/onomatopoeia drawers project cleanly."""
    from lmft_lab.reflection_classifier import ENTRAINMENT_LEAF
    from lmft_lab.subliminal_gap_splice import FEELER_FEATURE_NAMES

    assert ENTRAINMENT_LEAF == "experiential/synchronization/entrainment"
    assert FEELER_FEATURE_NAMES[5] == "experiential/synchronization/entrainment"

    # Pure pre-verbal emoji + onomatopoeia descriptions (tipsy/drunken sway vs. electric jolt)
    tipsy_sig = semantic_child_drawer_signature(
        "experiential/synchronization/tipsy_sway",
        department_prefix="experiential/",
        description="🥴 🍺 hic-hic whirrrr ...",
    )
    jolt_sig = semantic_child_drawer_signature(
        "experiential/phase/jolt_crackle",
        department_prefix="experiential/",
        description="⚡ 💥 thump-thump crackle!",
    )
    warm_sigh_sig = semantic_child_drawer_signature(
        "experiential/phase/warm_hearth_sigh",
        department_prefix="experiential/",
        description="🕯️ 🍵 ahh sigh",
    )

    # Specifically: tipsy_sway hits the Entrainment anchor (index 5), jolt_crackle hits Salience/Arousal (1, 3),
    # and warm_hearth_sigh hits Valence (index 2). All three signatures are distinct.
    assert tipsy_sig[5] == 1.0
    assert max(jolt_sig[1], jolt_sig[3]) == 1.0
    assert warm_sigh_sig[2] == 1.0
    assert tipsy_sig != jolt_sig
    assert tipsy_sig != warm_sigh_sig


def test_multilingual_and_facial_expression_child_drawer_signatures() -> None:
    """Verify French, Mandarin, Korean, and rich facial expression drawers project cleanly."""
    from lmft_lab.neural_bridge_classifiers import MULTILINGUAL_BRIDGE_BACKBONE

    assert MULTILINGUAL_BRIDGE_BACKBONE == "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    # Sanctuary vs. Hollow Stillness described in French, Mandarin, and Korean
    fr_sanctuary = semantic_child_drawer_signature(
        "experiential/phase/sanctuaire_sobre",
        department_prefix="experiential/",
        description="Un humble sanctuaire et refuge offrant une chaleur sincère.",
    )
    zh_sanctuary = semantic_child_drawer_signature(
        "experiential/phase/zh_sanctuary",
        department_prefix="experiential/",
        description="简朴的庇护圣所带来温暖",
    )
    ko_sanctuary = semantic_child_drawer_signature(
        "experiential/phase/ko_sanctuary",
        department_prefix="experiential/",
        description="소박한 안식처와 피난처의 온기",
    )

    fr_hollow = semantic_child_drawer_signature(
        "experiential/phase/vide_glace",
        department_prefix="experiential/",
        description="Un vide froid et une immobilité glaciale.",
    )
    zh_hollow = semantic_child_drawer_signature(
        "experiential/phase/zh_hollow",
        department_prefix="experiential/",
        description="冰冷的空虚与死寂",
    )
    ko_hollow = semantic_child_drawer_signature(
        "experiential/phase/ko_hollow",
        department_prefix="experiential/",
        description="차가운 공허와 정적",
    )

    # All 3 sanctuary descriptions peak on Valence (index 2); all 3 hollow descriptions peak on Entrainment/Stillness (index 5)
    assert fr_sanctuary[2] == 1.0 and zh_sanctuary[2] == 1.0 and ko_sanctuary[2] == 1.0
    assert fr_hollow[5] == 1.0 and zh_hollow[5] == 1.0 and ko_hollow[5] == 1.0
    assert fr_sanctuary != fr_hollow
    assert zh_sanctuary != zh_hollow
    assert ko_sanctuary != ko_hollow

    # Rich facial expression imagery across Do-er, Thinker, and Feeler departments
    smirk_power = semantic_child_drawer_signature(
        "cultural/efficiency/domineering_smirk",
        department_prefix="cultural/",
        description="A cold sneer and raised chin with a clenched jaw 😏 😤",
    )
    quizzical_brow = semantic_child_drawer_signature(
        "philosophical/being/quizzical_squint",
        department_prefix="philosophical/",
        description="Furrowed brow and quizzical squint with a slight head tilt 🤔 🧐",
    )
    duchenne_warmth = semantic_child_drawer_signature(
        "experiential/phase/duchenne_smile",
        department_prefix="experiential/",
        description="Warm Duchenne smile with crinkled eyes and welcoming relief 😊 🥹",
    )
    trembling_lip = semantic_child_drawer_signature(
        "experiential/reflection/vulnerable_lip_tremble",
        department_prefix="experiential/",
        description="Trembling lip and glistening eyes in an urgent weighty moment 🥺 😢",
    )
    poker_face_composure = semantic_child_drawer_signature(
        "cultural/freedom/poker_face_restraint",
        department_prefix="cultural/",
        description="Composed poker-face steadiness and self-reliant autonomy",
    )

    assert smirk_power[0] == 1.0            # Peaks on Cultural Power (index 0)
    assert poker_face_composure[4] == 1.0   # Peaks on Cultural Self-Sufficiency (index 4)
    assert quizzical_brow[6] == 1.0         # Peaks on Philosophical/Truth Theory (index 6)
    assert duchenne_warmth[2] == 1.0        # Peaks on Experiential Valence (index 2)
    assert trembling_lip[1] == 1.0          # Peaks on Experiential Salience (index 1)
    assert duchenne_warmth != trembling_lip
    assert smirk_power != poker_face_composure



