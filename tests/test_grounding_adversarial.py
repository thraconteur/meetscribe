"""Adversarial tests for the verifier: feed it the kinds of lies an LLM actually tells."""
from meetscribe.grounding import verify_record
from meetscribe.schema import Segment, Transcript, UNSPECIFIED

LINES = [
    "Okay let's start with the login service.",                                   # 1
    "Priya, can you fix the login bug on Android?",                               # 2
    "Sure, I'll take it, I can finish it by Friday.",                             # 3
    "Next, the payments gateway costs about four hundred dollars a month.",       # 4
    "Rahul said he would benchmark the gateway.",                                 # 5
    "We agreed to adopt the new gateway in Q three.",                             # 6
    "Maybe we could try GraphQL later, but let's think about it.",                # 7
    "Someone needs to write the runbook, we didn't decide who.",                  # 8
    "Filler one about lunch plans.", "Filler two about parking.", "Filler three about weather.",
    "Filler four about holidays.", "Filler five about coffee.", "Filler six about travel.",
    "Filler seven about cricket.", "Filler eight about movies.", "Filler nine about books.",
    "Okay the demo is on Monday, wrap up.",                                       # 18
]


def tr():
    return Transcript([Segment(i + 1, i * 5.0, i * 5.0 + 5, t) for i, t in enumerate(LINES)])


def task(**kw):
    base = dict(task="Fix the login bug on Android", owner=None, owner_evidence=None, deadline=None,
                deadline_evidence=None, evidence="Priya, can you fix the login bug on Android", segments=[2])
    base.update(kw)
    return base


def run(**raw):
    return verify_record(raw, tr())


def test_owner_kept_only_when_quote_contains_name():
    ok = run(action_items=[task(owner="Priya", owner_evidence="Priya, can you fix the login bug on Android")])
    assert ok.action_items[0].owner == "Priya"
    lie = run(action_items=[task(owner="Priya", owner_evidence="Sure, I'll take it, I can finish it by Friday")])
    assert lie.action_items[0].owner == UNSPECIFIED          # quote exists but never names Priya


def test_owner_name_that_never_appears_is_dropped():
    r = run(action_items=[task(owner="Amit", owner_evidence="Priya, can you fix the login bug on Android")])
    assert r.action_items[0].owner == UNSPECIFIED


def test_owner_quote_far_from_task_is_dropped():
    # Real quote naming Rahul, but 17 segments away from the login task.
    t = task(owner="Rahul", owner_evidence="Rahul said he would benchmark the gateway")
    t["task"], t["evidence"], t["segments"] = "Write the demo wrap-up", "Okay the demo is on Monday, wrap up", [18]
    r = run(action_items=[t])
    assert r.action_items[0].owner == UNSPECIFIED


def test_deadline_must_be_in_its_quote():
    ok = run(action_items=[task(deadline="by Friday", deadline_evidence="I can finish it by Friday")])
    assert ok.action_items[0].deadline == "by Friday"
    lie = run(action_items=[task(deadline="next Tuesday", deadline_evidence="I can finish it by Friday")])
    assert lie.action_items[0].deadline == UNSPECIFIED
    lie2 = run(action_items=[task(deadline="by Thursday", deadline_evidence="I can finish it by Friday")])
    assert lie2.action_items[0].deadline == UNSPECIFIED


def test_deadline_quote_far_from_task_is_dropped():
    r = run(action_items=[task(deadline="on Monday", deadline_evidence="the demo is on Monday, wrap up")])
    assert r.action_items[0].deadline == UNSPECIFIED


def test_pronoun_owner_is_unspecified():
    r = run(action_items=[task(owner="I", owner_evidence="Sure, I'll take it")])
    assert r.action_items[0].owner == UNSPECIFIED


def test_fabricated_quote_removes_decision_and_task():
    r = run(decisions=[dict(decision="Adopt Kafka", evidence="We unanimously chose Kafka for streaming", segments=[6])],
            action_items=[task(evidence="Amit must migrate the whole database tonight", segments=[3])])
    assert not r.decisions and not r.action_items


def test_real_quote_but_invented_number_is_dropped():
    r = run(decisions=[dict(decision="Budget approved at 900 dollars a month",
                            evidence="the payments gateway costs about four hundred dollars a month", segments=[4])])
    assert not r.decisions                                   # quote is real, the 900 is invented


def test_correct_number_in_digits_survives():
    r = run(decisions=[dict(decision="Adopt the new gateway in Q3 at about 400 dollars a month",
                            evidence="We agreed to adopt the new gateway in Q three", segments=[6])])
    assert len(r.decisions) == 1


def test_distinct_tasks_with_overlapping_wording_are_both_kept():
    r = run(action_items=[
        task(task="Fix the login bug"),
        task(task="Fix the login bug on Android before Friday", deadline="by Friday",
             deadline_evidence="I can finish it by Friday"),
    ])
    assert len(r.action_items) == 2


def test_true_duplicates_are_merged():
    r = run(action_items=[task(task="Fix the login bug on Android"), task(task="Fix the login bug on android.")])
    assert len(r.action_items) == 1


def test_settled_proposal_not_listed_as_open_but_unsettled_one_is():
    r = run(decisions=[dict(decision="Adopt the new gateway in Q3",
                            evidence="We agreed to adopt the new gateway in Q three", segments=[6])],
            open_items=[dict(item="Adopt the new gateway in Q3", kind="proposal",
                             evidence="We agreed to adopt the new gateway in Q three", segments=[6]),
                        dict(item="Try GraphQL later", kind="proposal",
                             evidence="Maybe we could try GraphQL later, but let's think about it", segments=[7])])
    assert [o.item for o in r.open_items] == ["Try GraphQL later"]


def test_proposal_never_becomes_task_owner_from_unlabelled_speaker():
    r = run(action_items=[task(task="Write the runbook", owner="Priya",
                               owner_evidence="Priya, can you fix the login bug on Android",
                               evidence="Someone needs to write the runbook, we didn't decide who", segments=[8])])
    # Priya's quote is 6 segments away and about a different task; the model must not get away with it
    assert r.action_items[0].owner == UNSPECIFIED


def test_hedged_quote_cannot_back_a_decision_it_is_demoted_to_open_item():
    r = run(decisions=[dict(decision="Adopt GraphQL for the public API",
                            evidence="Maybe we could try GraphQL later, but let's think about it", segments=[7])])
    assert not r.decisions
    assert [o.kind for o in r.open_items] == ["proposal"]
    assert any("Moved" in line for line in r.verification_log)


def test_agreed_quote_with_a_hedge_word_is_still_a_decision():
    r = run(decisions=[dict(decision="Adopt the new gateway in Q3",
                            evidence="We agreed to adopt the new gateway in Q three", segments=[6])])
    assert len(r.decisions) == 1 and not r.open_items


# ---- minutes / summary: no leaked internals, no invented numbers ------------------------------------------
from meetscribe.grounding import strip_segment_refs  # noqa: E402


def test_internal_segment_refs_are_stripped_from_all_prose():
    raw = dict(summary="We reviewed the gateway (S4‑S6). Next steps agreed [S18].",
               minutes=[dict(topic="Gateway (S4)", points=["Costs about four hundred dollars (S4‑S5)", "Rahul benchmarks it S5‑S6."])])
    r = run(**raw)
    text = r.summary + r.minutes[0].topic + " ".join(r.minutes[0].points)
    assert "(S" not in text and "[S" not in text and "S5‑S6" not in text
    assert r.minutes[0].points[0] == "Costs about four hundred dollars"


def test_a_reference_the_meeting_really_said_is_kept():
    assert strip_segment_refs("Move logs to the bucket (S3) next week", "we store logs in S3 today") == \
        "Move logs to the bucket (S3) next week"


def test_minutes_point_with_invented_number_is_dropped_but_correct_ones_stay():
    r = run(minutes=[dict(topic="Costs", points=["Gateway costs 400 dollars a month", "Gateway costs 900 dollars a month"])])
    assert r.minutes[0].points == ["Gateway costs 400 dollars a month"]
    assert any("minutes point" in line for line in r.verification_log)


def test_summary_sentence_with_invented_number_is_dropped():
    r = run(summary="The team discussed the gateway. It will cost 900 dollars a month. Rahul will benchmark it.")
    assert "900" not in r.summary and "gateway" in r.summary and "benchmark" in r.summary


def test_task_text_with_leaked_ref_is_not_wrongly_removed_by_number_check():
    r = run(action_items=[task(task="Fix the login bug on Android (S2)")])
    assert len(r.action_items) == 1 and "(S2)" not in r.action_items[0].task
