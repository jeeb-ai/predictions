"""First-receipt grading, pending/void/tie rules and revision-safe official outcomes."""

import hashlib
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest
from test_platform_automation import forecast

from predictions.config import initialize_research, load_config
from predictions.prospective import ForecastJournal, connect
from predictions.tracking import OutcomeJournal, normalize_result, reconcile, report_for

AT = pd.Timestamp("2026-10-30T12:00:00Z")
RECEIVED = AT - pd.Timedelta(days=4)
KICKOFF = AT - pd.Timedelta(days=2)


def receipt(sport="NBA", identity="f1", game="game1", p=0.6):
    value = forecast(sport, identity, RECEIVED)
    value["game_id"] = game
    value["kickoff"] = KICKOFF.isoformat()
    value["home_win_probability"] = p
    value["away_win_probability"] = 1 - p
    return value


def result(
    tmp_path, sport="NBA", home=110, away=100, *, at=AT, state="final", kind=None, game="game1"
):
    raw = receipt(sport)
    value = {
        "sport": sport,
        "game_id": game,
        "home_team": raw["home_team"],
        "away_team": raw["away_team"],
        "kickoff": KICKOFF.isoformat(),
        "status": state,
        "home_score": home,
        "away_score": away,
        "outcome_type": kind or ("REG" if sport == "NHL" else "FULL_OT"),
        "observed_at": at.isoformat(),
    }
    body = json.dumps(value).encode()
    sha = hashlib.sha256(body).hexdigest()
    path = tmp_path / (sha + ".json")
    path.write_bytes(body)
    value["source"] = {
        "sha256": sha,
        "path": str(path),
        "url": "https://example.org/explicit-synthetic-official-fixture",
        "provider": "Synthetic fixture, not a real outcome",
        "classification": "official_league_data",
        "fetched_at": at.isoformat(),
    }
    return value


@pytest.fixture
def setup(tmp_path):
    config = tmp_path / "research.json"
    initialize_research(config)
    c = load_config(config)
    journal = OutcomeJournal(tmp_path / "results.sqlite")
    return c, journal


def test_first_receipt_wins_pending_never_losses_and_later_model_cannot_replace(setup, tmp_path):
    config, journal = setup
    first = receipt()
    later = receipt(identity="later", p=0.1)
    later["model_version"] = "nba-v2"
    db = ForecastJournal(journal.path)
    db.capture("NBA", [first], received_at=RECEIVED)
    db.capture("NBA", [later], received_at=RECEIVED + pd.Timedelta(hours=1))
    pending = report_for(journal, "NBA", as_of=AT)
    assert (
        pending["overall"]["accuracy"] is None
        and pending["overall"]["losses"] == 0
        and pending["overall"]["pending"] == 1
    )
    journal.append(result(tmp_path), as_of=AT)
    graded = report_for(journal, "NBA", as_of=AT)
    assert graded["overall"]["wins"] == 1 and graded["overall"]["losses"] == 0
    assert graded["overall"]["brier"] == pytest.approx(0.16) and graded["overall"][
        "log_loss"
    ] == pytest.approx(-np.log(0.6))
    assert graded["overall"]["ece"] == pytest.approx(0.4)
    assert sum(b["count"] for b in graded["overall"]["calibration"]) == 1
    assert list(graded["by_model_version"]) == ["NBA-v1"]
    assert graded["history"][0]["forecast_id"] == "f1" and graded["retained_receipts"] == 2
    before = report_for(journal, "NBA", as_of=RECEIVED)
    assert before["overall"]["graded"] == 0


def test_postgame_import_cannot_create_performance(setup, tmp_path):
    _, journal = setup
    ForecastJournal(journal.path).capture("NBA", [receipt()], received_at=AT)
    journal.append(result(tmp_path), as_of=AT)
    report = report_for(journal, "NBA", as_of=AT)
    assert (
        report["late_imports"] == 1
        and report["overall"]["predicted"] == 0
        and report["overall"]["accuracy"] is None
    )


def test_nfl_tie_is_not_a_winner_loss_but_scores_are_graded(setup, tmp_path):
    _, journal = setup
    row = receipt("NFL")
    # This synthetic verified score model is intentionally independent of the winner probability.
    mass = np.zeros((30, 30))
    mass[24, 20] = 1
    from predictions.league_model import summarize

    path = tmp_path / "mass.npy"
    np.save(path, mass, allow_pickle=False)
    row["score_prediction"] = {
        "as_of": row["as_of"],
        "generated_at": row["generated_at"],
        "distribution": {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
        "summary": summarize(mass),
    }
    ForecastJournal(journal.path).capture("NFL", [row], received_at=RECEIVED)
    journal.append(result(tmp_path, "NFL", 21, 21), as_of=AT)
    report = report_for(journal, "NFL", as_of=AT)
    m = report["overall"]
    assert (
        m["ties"] == 1
        and m["picked_games"] == 0
        and m["accuracy"] is None
        and m["score_games"] == 1
    )
    assert m["score_mae"] == pytest.approx(2) and m["margin_mae"] == 4 and m["total_mae"] == 2
    assert report["recorded_lines"] and all(
        x["market_status"].startswith("PASS") for x in report["recorded_lines"]
    )


@pytest.mark.parametrize(
    "state,expected",
    [
        ("cancelled", "cancelled"),
        ("postponed", "pending"),
        ("incomplete", "pending"),
        ("disputed", "review"),
    ],
)
def test_nonfinal_results_do_not_become_losses(setup, tmp_path, state, expected):
    _, journal = setup
    ForecastJournal(journal.path).capture("NBA", [receipt()], received_at=RECEIVED)
    journal.append(result(tmp_path, state=state, home=None, away=None), as_of=AT)
    report = report_for(journal, "NBA", as_of=AT)
    assert report["history"][0]["status"] == expected
    assert report["overall"]["losses"] == 0 and report["overall"]["accuracy"] is None


def test_correction_requires_later_confirmation_and_stale_source_cannot_undo(setup, tmp_path):
    _, journal = setup
    ForecastJournal(journal.path).capture("NBA", [receipt()], received_at=RECEIVED)
    first = result(tmp_path)
    assert journal.append(first, as_of=AT) == "final"
    assert journal.append(first, as_of=AT) == "unchanged"
    changed = result(tmp_path, home=99, away=100, at=AT + pd.Timedelta(minutes=1))
    assert journal.append(changed, as_of=AT + pd.Timedelta(minutes=1)) == "disputed"
    assert report_for(journal, "NBA", as_of=AT + pd.Timedelta(minutes=1))["overall"]["graded"] == 0
    early = result(tmp_path, home=99, away=100, at=AT + pd.Timedelta(minutes=2))
    assert journal.append(early, as_of=AT + pd.Timedelta(minutes=2)) == "awaiting_confirmation"
    confirmed = result(tmp_path, home=99, away=100, at=AT + pd.Timedelta(minutes=7))
    assert journal.append(confirmed, as_of=AT + pd.Timedelta(minutes=7)) == "final"
    assert journal.append(first, as_of=AT + pd.Timedelta(minutes=8)) == "stale_source"
    report = report_for(journal, "NBA", as_of=AT + pd.Timedelta(minutes=8))
    assert (
        report["overall"]["losses"] == 1
        and len(journal.observations(AT + pd.Timedelta(minutes=8))) == 3
    )
    assert report_for(journal, "NBA", as_of=AT)["overall"]["wins"] == 1


def test_schedule_change_requires_review_and_future_result_is_rejected(setup, tmp_path):
    _, journal = setup
    ForecastJournal(journal.path).capture("NBA", [receipt()], received_at=RECEIVED)
    changed = result(tmp_path)
    changed["kickoff"] = (KICKOFF + pd.Timedelta(hours=1)).isoformat()
    journal.append(changed, as_of=AT)
    assert report_for(journal, "NBA", as_of=AT)["overall"]["review"] == 1
    with pytest.raises(ValueError, match="future"):
        journal.append(result(tmp_path, at=AT + pd.Timedelta(days=1)), as_of=AT)


@pytest.mark.parametrize(
    "kind,home,away,reg", [("REG", 4, 1, (4, 1)), ("OT", 3, 2, (2, 2)), ("SO", 1, 2, (1, 1))]
)
def test_hockey_scopes_are_distinct_and_full_winner_is_graded(
    setup, tmp_path, kind, home, away, reg
):
    _, journal = setup
    ForecastJournal(journal.path).capture("NHL", [receipt("NHL")], received_at=RECEIVED)
    value = normalize_result(result(tmp_path, "NHL", home, away, kind=kind))
    assert (
        value["content"]["regulation_home_score"],
        value["content"]["regulation_away_score"],
    ) == reg
    journal.append(result(tmp_path, "NHL", home, away, kind=kind), as_of=AT)
    assert report_for(journal, "NHL", as_of=AT)["overall"]["graded"] == 1
    if kind == "SO":
        assert report_for(journal, "NHL", as_of=AT)["overall"]["losses"] == 1


def test_invalid_nhl_scope_and_wrong_result_provenance_fail(tmp_path):
    with pytest.raises(ValueError, match="scope"):
        normalize_result(result(tmp_path, "NHL", 4, 1, kind="OT"))
    value = result(tmp_path)
    value["source"]["classification"] = "illustrative"
    with pytest.raises(ValueError, match="provenance"):
        normalize_result(value)


def test_reconcile_deduplicates_games_and_keeps_unaffected_sport_results(tmp_path):
    configpath = tmp_path / "research.json"
    initialize_research(configpath)
    config = load_config(configpath)
    path = tmp_path / "data/research/prospective.sqlite"
    db = ForecastJournal(path)
    db.capture("NBA", [receipt()], received_at=RECEIVED)
    db.capture("NFL", [receipt("NFL")], received_at=RECEIVED)

    def feed(sport, targets, root):
        if sport == "NFL":
            raise ConnectionError("offline")
        return [result(tmp_path)]

    a = reconcile(config, "NBA", fetcher=feed, clock=lambda: AT)
    b = reconcile(config, "NBA", fetcher=feed, clock=lambda: AT)
    c = reconcile(config, "NFL", fetcher=feed, clock=lambda: AT)
    assert a["overall"]["graded"] == b["overall"]["graded"] == 1
    assert (
        c["overall"]["pending"] == 1
        and c["overall"]["losses"] == 0
        and c["reconciliation"]["status"] == "partial"
    )
    with connect(path, readonly=True) as sql:
        assert sql.execute("SELECT count(*) FROM result_observations").fetchone()[0] == 1
        assert (
            sql.execute('SELECT count(*) FROM prospective_metrics WHERE sport="NBA"').fetchone()[0]
            == 1
        )
    with connect(path) as sql:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            sql.execute("DELETE FROM result_observations")


def test_weaker_stale_archive_does_not_retract_official_final(setup, tmp_path):
    _, journal = setup
    first = result(tmp_path)
    journal.append(first, as_of=AT)
    stale = result(tmp_path, state="scheduled", home=None, away=None, at=AT + pd.Timedelta(hours=1))
    stale["source"]["classification"] = "official_stats_archive"
    assert journal.append(stale, as_of=AT + pd.Timedelta(hours=1)) == "stale_archive_nonfinal"
    assert journal.observations(AT + pd.Timedelta(hours=1))[-1]["content"]["status"] == "final"


def test_even_probability_has_no_invented_pick_but_gets_proper_probability_scores(setup, tmp_path):
    _, journal = setup
    ForecastJournal(journal.path).capture("NBA", [receipt(p=0.5)], received_at=RECEIVED)
    journal.append(result(tmp_path), as_of=AT)
    report = report_for(journal, "NBA", as_of=AT)
    assert report["overall"]["picked_games"] == 0 and report["overall"]["accuracy"] is None
    assert report["overall"]["probability_games"] == 1 and report["overall"]["brier"] == 0.25


def test_capture_official_reconciliation_authenticated_dashboard_end_to_end(tmp_path, monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    configpath = tmp_path / "research.json"
    initialize_research(configpath)
    config = load_config(configpath)
    journal = ForecastJournal(Path(config["paths"]["state_dir"]) / "prospective.sqlite")
    for sport in ["NFL", "NBA", "NHL"]:
        journal.capture(sport, [receipt(sport)], received_at=RECEIVED)
        journal.capture(sport, [receipt(sport, identity="later", p=0.1)], received_at=RECEIVED)
        report = reconcile(
            config,
            sport,
            fetcher=lambda league, targets, root: [result(tmp_path, league, 4, 1, kind="REG")],
            clock=lambda: AT,
        )
        assert report["history"][0]["home_win_probability"] == 0.6
        assert report["overall"]["wins"] == 1
    monkeypatch.setenv("NFL_RESEARCH_CONFIG", str(configpath))
    before = {p: p.read_bytes() for p in journal.path.parent.glob("*-prospective.json")}
    app = AppTest.from_string("from predictions.dashboard import main\nmain()").run()
    assert not app.metric  # No private performance leaks through the login screen.
    app.text_input[0].set_value(Path(config["dashboard"]["token_file"]).read_text().strip())
    app.button[0].click().run(timeout=30)
    assert not app.exception
    accuracy = [m.value for m in app.metric if m.label == "Correct winners"]
    assert accuracy == ["100.0%"] * 3  # Explicit synthetic fixtures, never production evidence.
    assert any(x.label == "Prospective accuracy — future games" for x in app.expander)
    assert before == {p: p.read_bytes() for p in before}  # UI is read-only.
    app.selectbox(key="prospective_period").select("Last 7 days").run()
    assert not app.exception


def test_result_poll_interval_and_no_grading_before_conservative_availability(tmp_path):
    from pathlib import Path

    configpath = tmp_path / "research.json"
    initialize_research(configpath)
    config = load_config(configpath)
    ForecastJournal(Path(config["paths"]["state_dir"]) / "prospective.sqlite").capture(
        "NBA", [receipt()], received_at=RECEIVED
    )
    calls = []

    def feed(sport, targets, root):
        calls.append(targets)
        return [result(tmp_path)]

    early = reconcile(config, "NBA", fetcher=feed, clock=lambda: KICKOFF + pd.Timedelta(hours=3))
    assert early["overall"]["pending"] == 1 and not calls
    report = reconcile(config, "NBA", fetcher=feed, clock=lambda: AT)
    repeat = reconcile(config, "NBA", fetcher=feed, clock=lambda: AT + pd.Timedelta(minutes=1))
    assert report["overall"]["graded"] == repeat["overall"]["graded"] == 1
    assert len(calls) == 1 and not repeat["reconciliation"]["official_fetch_requested"]


def test_replayed_source_cannot_independently_confirm_a_correction(setup, tmp_path):
    _, journal = setup
    journal.append(result(tmp_path), as_of=AT)
    changed = result(tmp_path, home=99, at=AT + pd.Timedelta(minutes=1))
    assert journal.append(changed, as_of=AT + pd.Timedelta(minutes=1)) == "disputed"
    replay = {**changed, "observed_at": (AT + pd.Timedelta(minutes=7)).isoformat()}
    assert journal.append(replay, as_of=AT + pd.Timedelta(minutes=7)) == "awaiting_confirmation"


def test_archive_restore_cannot_downgrade_the_confirmed_official_source(setup, tmp_path):
    _, journal = setup
    journal.append(result(tmp_path), as_of=AT)
    changed = result(tmp_path, home=99, at=AT + pd.Timedelta(minutes=1))
    changed["source"]["classification"] = "official_stats_archive"
    assert journal.append(changed, as_of=AT + pd.Timedelta(minutes=1)) == "disputed"
    restored = result(tmp_path, at=AT + pd.Timedelta(minutes=2))
    restored["source"]["classification"] = "official_stats_archive"
    assert journal.append(restored, as_of=AT + pd.Timedelta(minutes=2)) == "final"
    assert (
        journal.observations(AT + pd.Timedelta(minutes=2))[-1]["last_confirmed_source_class"]
        == "official_league_data"
    )
    nonfinal = result(tmp_path, state="scheduled", at=AT + pd.Timedelta(minutes=3))
    nonfinal["source"]["classification"] = "official_stats_archive"
    assert journal.append(nonfinal, as_of=AT + pd.Timedelta(minutes=3)) == "stale_archive_nonfinal"
