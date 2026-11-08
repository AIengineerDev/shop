from refunds.config import PolicyConfig
from refunds.model import Outcome, Signals
from refunds.policy import DEFAULT_RULES, abuse_score_bounded, decide
from tests.refunds.conftest import NOW, order, request, view

P = PolicyConfig()


def run(amount="40.00", score=None, paid="500.00"):
    return decide(order(paid=paid), request(amount=amount), view(paid=paid), NOW, P, Signals(abuse_score=score))


def test_missing_score_is_a_noop():
    assert run().outcome is Outcome.APPROVE and run(score=None).reasons == ["eligible"]


def test_high_score_escalates_with_explanation():
    d = run(score=0.93)
    assert d.outcome is Outcome.ESCALATE and d.reasons == ["abuse_score:0.93>=0.80"]


def test_mid_score_halves_auto_limit():
    assert run("100.00", score=0.6).outcome is Outcome.APPROVE
    d = run("150.00", score=0.6)
    assert d.outcome is Outcome.ESCALATE
    assert d.reasons == ["abuse_score_lowered_limit:0.60>=0.50,limit=100.00 USD"]


def test_low_score_changes_nothing():
    assert run("150.00", score=0.1) == run("150.00")


def test_rule_never_approves():
    """Bounded: whatever the score, the rule can only add ESCALATE verdicts."""
    from refunds.policy import Context
    for s in (0.0, 0.49, 0.5, 0.79, 0.8, 1.0):
        ctx = Context(order(), request(amount="190.00"), view(), NOW, Signals(abuse_score=s), P, request(amount="190.00").amount)
        v = abuse_score_bounded(ctx)
        assert v is None or v.outcome is Outcome.ESCALATE
    assert abuse_score_bounded in DEFAULT_RULES
