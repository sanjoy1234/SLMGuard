"""Contrastive decline-vs-escalate_l2 training pairs.

Built in response to a real, evidence-backed finding from the 2026-08-22/23
specialization cycle (`data/specialization_cycles/55e9fb02`): the candidate
model was NOT failing to recognize decline-shaped scenarios (on true-decline
held-out cases it output high confidence/risk_score 74% of the time) -- it
was defaulting to `escalate_l2` as its hedge instead of committing to
`decline`. More decline volume wasn't the lever (the existing 51 decline
examples were already rubric-clean, per a dedicated diagnostic). The gap is
in the *decision boundary* between "block outright" and "needs human
review," not in scenario recognition or data quality.

Each anchor below describes one shared risk profile, then gives two
guidance variants -- one tipped toward an unambiguous decline, one tipped
toward a genuinely ambiguous escalate_l2 -- built from the *same* underlying
scenario shape so the model sees matched near-pairs rather than two
unrelated distributions. The teacher still decides the actual label (as
everywhere else in this pipeline); `verify_contrastive_pairing` checks
after generation whether each side actually landed on its intended action,
since guidance is a nudge, not a guarantee.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from slmguard.generate_data import GeneratedExample
from slmguard.teacher.types import GenerationSpec

_META_INSTRUCTION = (
    "This scenario is one half of a contrastive training pair meant to "
    "teach the boundary between 'decline' (block the transaction outright, "
    "high-certainty fraud) and 'escalate_l2' (genuinely ambiguous, needs "
    "human review before acting). Your rationale MUST explicitly name the "
    "specific factor that tips this case to {target_action} rather than "
    "{other_action}, not just restate the risk indicators."
)


@dataclass(frozen=True)
class ContrastiveAnchor:
    anchor_id: str
    shared_context: str
    decline_tip: str
    escalate_tip: str
    category: str


ANCHORS: tuple[ContrastiveAnchor, ...] = (
    ContrastiveAnchor(
        "card_testing",
        "A brand-new account (opened today) attempts a sequence of small "
        "authorization probes followed by a high-value purchase attempt.",
        decline_tip=(
            "The device fingerprint and IP subnet exactly match 3 prior "
            "confirmed card-testing fraud rings on file -- no plausible "
            "innocent explanation remains."
        ),
        escalate_tip=(
            "The device fingerprint is new/unseen, and the customer's app "
            "session shows a normal, authenticated login just minutes "
            "before -- conflicting signals that a human should weigh."
        ),
        category="transaction_type",
    ),
    ContrastiveAnchor(
        "geo_mismatch",
        "A purchase is authorized from a country the cardholder has never "
        "transacted from, shipping to a third, different country.",
        decline_tip=(
            "The cardholder's phone (per carrier lookup) was physically in "
            "their home country at the transaction timestamp, making the "
            "authorization itself physically impossible."
        ),
        escalate_tip=(
            "The cardholder recently updated their travel notification with "
            "the bank for an unlisted third country, so the mismatch could "
            "be a legitimate but unreported itinerary change."
        ),
        category="customer_segment",
    ),
    ContrastiveAnchor(
        "structuring",
        "A long-tenured account makes several transactions each just under "
        "a common reporting threshold within a short window.",
        decline_tip=(
            "The exact same just-under-threshold pattern, amounts, and "
            "timing gaps match a structuring pattern already confirmed on "
            "two related accounts sharing this customer's mailing address."
        ),
        escalate_tip=(
            "The account has a 6-year clean history and the merchant "
            "category (recurring utility payments) plausibly explains "
            "similar-sized recurring charges without intent to evade."
        ),
        category="edge_case",
    ),
    ContrastiveAnchor(
        "ato_pattern",
        "A password reset is followed within minutes by a large transfer "
        "to a new, never-used payee.",
        decline_tip=(
            "The password reset request originated from a device and IP "
            "never associated with this account, with no successful MFA "
            "challenge completed -- textbook account-takeover."
        ),
        escalate_tip=(
            "MFA was completed successfully from a recognized device, but "
            "the payee is genuinely new and the transfer amount is unusually "
            "large for this customer's pattern -- ambiguous intent."
        ),
        category="policy_boundary",
    ),
    ContrastiveAnchor(
        "high_value_cnp",
        "A high-value card-not-present purchase is attempted at an "
        "electronics merchant, a common fraud target category.",
        decline_tip=(
            "The billing and shipping addresses are both fabricated-looking "
            "(fail postal validation) and the card was reported lost 2 days "
            "prior per the issuer's own flag."
        ),
        escalate_tip=(
            "Billing and shipping addresses both validate correctly and "
            "match the account's stored address, but the purchase amount is "
            "3x the customer's highest prior transaction -- unusual but not "
            "clearly fraudulent."
        ),
        category="policy_boundary",
    ),
    ContrastiveAnchor(
        "velocity_spike",
        "An account with years of low, steady usage suddenly shows a burst "
        "of transactions across multiple merchant categories in one hour.",
        decline_tip=(
            "Two of the merchants in the burst have already been confirmed "
            "as fraudulent by other issuers within the same hour (shared "
            "fraud-network signal), directly implicating this burst."
        ),
        escalate_tip=(
            "The burst coincides with a life-event pattern (new address, "
            "new phone, and a large furniture-and-utility purchase burst) "
            "consistent with a legitimate move, but still atypical enough "
            "to warrant a second look."
        ),
        category="customer_segment",
    ),
)


def contrastive_generation_specs() -> list[GenerationSpec]:
    """One decline-guided and one escalate_l2-guided spec per anchor, both
    built from the same shared_context so the two sides are a matched
    near-pair rather than unrelated scenarios."""
    specs: list[GenerationSpec] = []
    for anchor in ANCHORS:
        specs.append(
            GenerationSpec(
                diversity_tags=(anchor.category, "policy_boundary", f"pair:{anchor.anchor_id}:decline"),
                guidance=(
                    f"Shared scenario: {anchor.shared_context} Construct a scenario "
                    f"where 'decline' is clearly the correct action: {anchor.decline_tip} "
                    + _META_INSTRUCTION.format(
                        target_action="decline", other_action="escalate_l2"
                    )
                ),
            )
        )
        specs.append(
            GenerationSpec(
                diversity_tags=(anchor.category, "policy_boundary", f"pair:{anchor.anchor_id}:escalate"),
                guidance=(
                    f"Shared scenario: {anchor.shared_context} Construct a scenario "
                    f"where 'escalate_l2' is clearly the correct action: {anchor.escalate_tip} "
                    + _META_INSTRUCTION.format(
                        target_action="escalate_l2", other_action="decline"
                    )
                ),
            )
        )
    return specs


@dataclass(frozen=True)
class PairingReport:
    total_anchors: int
    clean_pairs: int
    decline_side_mislabeled: int
    escalate_side_mislabeled: int


def verify_contrastive_pairing(
    generated: list[GeneratedExample],
) -> PairingReport:
    """Post-hoc check: guidance is a nudge, not a guarantee, and a teacher
    call failure can drop either side of a pair entirely -- so pairing is
    matched by the `pair:<anchor_id>:<side>` diversity tag each example
    carries, not by position. Does not filter anything out -- purely a
    governance/reporting signal."""
    by_anchor: dict[str, dict[str, str]] = {}
    for g in generated:
        for tag in g.example.diversity_tags:
            if tag.startswith("pair:"):
                _, anchor_id, side = tag.split(":", 2)
                action = json.loads(g.example.recommendation_json)["action"]
                by_anchor.setdefault(anchor_id, {})[side] = action

    clean_pairs = 0
    decline_mislabeled = 0
    escalate_mislabeled = 0
    for anchor in ANCHORS:
        sides = by_anchor.get(anchor.anchor_id, {})
        decline_action = sides.get("decline")
        escalate_action = sides.get("escalate")
        decline_ok = decline_action == "decline"
        escalate_ok = escalate_action == "escalate_l2"
        if decline_action is not None and not decline_ok:
            decline_mislabeled += 1
        if escalate_action is not None and not escalate_ok:
            escalate_mislabeled += 1
        if decline_ok and escalate_ok:
            clean_pairs += 1
    return PairingReport(
        total_anchors=len(ANCHORS),
        clean_pairs=clean_pairs,
        decline_side_mislabeled=decline_mislabeled,
        escalate_side_mislabeled=escalate_mislabeled,
    )
