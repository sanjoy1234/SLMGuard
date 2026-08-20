"""The policy rules engine: declarative, versioned rules the control plane
applies to a model's recommendation after schema validation, per
docs/technical-build-plan-v5.md "Control Plane Failure Modes" -- "Model
output is schema-valid but violates a policy rule -> The policy engine
overrides the model's recommendation... The model is never trusted merely
because its output parsed."

Rules are data, not code: each `PolicyRule` is a `kind` (from a fixed,
reviewable set) plus a numeric `threshold`, not an arbitrary expression or
callback. Adding or tuning a rule means editing a `PolicyRuleSet` literal,
never writing new control-flow -- that's what "declarative" buys here, and
it's also why there's no eval()/exec() or expression parser: the rule space
is deliberately small and closed.

Not behind a swappable backend interface like ModelBackend/AuditStore --
there's no local-vs-production infrastructure split to abstract over here,
just pure computation over a `Recommendation`'s own fields. Versioning
(`PolicyRuleSet.version`) is what makes this auditable over time, not a
backend swap.

Known limitation, honestly stated: rules only see the `Recommendation`'s own
fields (action, risk_score, confidence) -- there is no structured Alert
schema yet (transaction type, channel, amount, ...), so a rule like "certain
transaction types always escalate" (named explicitly in the build plan)
isn't implementable until that structured intake exists. `policy_flags` on
the recommendation are the model's own self-reported claims, not verified
facts, and are deliberately not used as a rule input for that reason -- a
model that wanted to evade a flag-based rule could just omit the flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from slmguard.schema import Action, Recommendation


class RuleKind(str, Enum):
    CONFIDENCE_BELOW_FORCES_ESCALATION = "confidence_below_forces_escalation"
    HIGH_RISK_CANNOT_APPROVE = "high_risk_cannot_approve"
    LOW_RISK_CANNOT_DECLINE = "low_risk_cannot_decline"


@dataclass(frozen=True)
class PolicyRule:
    rule_id: str
    kind: RuleKind
    threshold: float
    description: str


@dataclass(frozen=True)
class PolicyRuleSet:
    version: str
    rules: tuple[PolicyRule, ...]


@dataclass(frozen=True)
class PolicyDecision:
    """The control plane's final say, and the paper trail for it. A rule
    violation always forces escalate_l2 -- policy overrides never approve or
    decline on the model's behalf, they only ever route to a human/L2
    reviewer, since a rule firing means the model's own call can't be
    trusted as-is."""

    policy_version: str
    original_action: Action
    final_action: Action
    overridden: bool
    violated_rule_ids: tuple[str, ...]


DEFAULT_POLICY_RULESET = PolicyRuleSet(
    version="policy-v1",
    rules=(
        PolicyRule(
            rule_id="forced_escalation_low_confidence",
            kind=RuleKind.CONFIDENCE_BELOW_FORCES_ESCALATION,
            threshold=0.5,
            description=(
                "Confidence below 0.5 always forces escalation to L2, "
                "regardless of the model's stated action -- matches the build "
                "plan's own stated example of a forced-escalation rule."
            ),
        ),
        PolicyRule(
            rule_id="no_silent_approve_high_risk",
            kind=RuleKind.HIGH_RISK_CANNOT_APPROVE,
            threshold=0.8,
            description=(
                "A risk_score >= 0.8 can never be silently approved -- the "
                "model is never trusted merely because its output parsed, and "
                "a high self-reported risk paired with 'approve' is exactly "
                "the kind of output that must not reach a customer unreviewed."
            ),
        ),
        PolicyRule(
            rule_id="no_low_risk_decline",
            kind=RuleKind.LOW_RISK_CANNOT_DECLINE,
            threshold=0.3,
            description=(
                "A risk_score < 0.3 cannot be declined outright -- an "
                "apparently low-risk case being declined is also a business "
                "risk (wrongful denial) and gets routed to a human instead of "
                "auto-applied."
            ),
        ),
    ),
)


def _rule_violated(rule: PolicyRule, recommendation: Recommendation) -> bool:
    if rule.kind == RuleKind.CONFIDENCE_BELOW_FORCES_ESCALATION:
        return (
            recommendation.confidence < rule.threshold
            and recommendation.action != Action.ESCALATE_L2
        )
    if rule.kind == RuleKind.HIGH_RISK_CANNOT_APPROVE:
        return recommendation.risk_score >= rule.threshold and recommendation.action == Action.APPROVE
    if rule.kind == RuleKind.LOW_RISK_CANNOT_DECLINE:
        return recommendation.risk_score < rule.threshold and recommendation.action == Action.DECLINE
    raise ValueError(f"Unknown rule kind: {rule.kind}")


def apply_policy(
    recommendation: Recommendation, ruleset: PolicyRuleSet = DEFAULT_POLICY_RULESET
) -> PolicyDecision:
    """Apply every rule in `ruleset` to `recommendation`. Any violation forces
    escalate_l2 and is recorded -- multiple simultaneous violations are all
    reported, not just the first one found."""
    violated = tuple(rule.rule_id for rule in ruleset.rules if _rule_violated(rule, recommendation))
    overridden = bool(violated)
    return PolicyDecision(
        policy_version=ruleset.version,
        original_action=recommendation.action,
        final_action=Action.ESCALATE_L2 if overridden else recommendation.action,
        overridden=overridden,
        violated_rule_ids=violated,
    )
