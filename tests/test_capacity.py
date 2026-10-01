"""End-to-end tests for per-point capacity (同时接待名额) and its waitlist."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def iso(dt):
    return dt.isoformat().replace("+00:00", "Z")


def at(hour, minute=0, day_offset=1):
    day = (datetime.now(UTC) + timedelta(days=day_offset)).date()
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=UTC)


def create_point(client, capacity=None, tz="UTC"):
    body = {"name": "容量测试点", "timezone": tz}
    if capacity is not None:
        body["capacity"] = capacity
    r = client.post("/points", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def open_point(client, capacity=None, tz="UTC", weekly=None):
    point = create_point(client, capacity=capacity, tz=tz)
    tomorrow = (datetime.now(UTC) + timedelta(days=1)).date()
    r = client.post(
        f"/points/{point['id']}/rules",
        json={
            "expected_version": 0,
            "rules": {
                "weekly": weekly
                if weekly is not None
                else [
                    {
                        "weekday": tomorrow.isoweekday(),
                        "start_time": "08:00",
                        "end_time": "12:00",
                    }
                ],
                "closed_dates": [],
                "makeup": [],
            },
        },
    )
    assert r.status_code == 201, r.text
    return point


def reserve(client, pid, key, start, end):
    return client.post(
        f"/points/{pid}/reservations",
        json={"idempotency_key": key, "start": iso(start), "end": iso(end)},
    )


def waitlist(client, pid, key, start, end):
    return client.post(
        f"/points/{pid}/waitlist",
        json={"idempotency_key": key, "start": iso(start), "end": iso(end)},
    )


def status(client, pid, wid, token):
    return client.get(
        f"/points/{pid}/waitlist/{wid}", params={"query_token": token}
    )


def cancel(client, pid, rid, token):
    return client.post(
        f"/points/{pid}/reservations/{rid}/cancel",
        json={"cancel_token": token},
    )


# ---------------------------------------------------------------------------
# Capacity on point read / update
# ---------------------------------------------------------------------------


def test_new_and_old_points_default_to_capacity_one(client):
    body = create_point(client)
    assert body["capacity"] == 1
    fetched = client.get(f"/points/{body['id']}").json()
    assert fetched["capacity"] == 1


def test_create_point_accepts_capacity_between_1_and_20(client):
    body = create_point(client, capacity=20)
    assert body["capacity"] == 20
    assert client.get(f"/points/{body['id']}").json()["capacity"] == 20


def test_create_point_rejects_capacity_out_of_range(client):
    for bad in (0, 21, -1):
        r = client.post(
            "/points", json={"name": "x", "timezone": "UTC", "capacity": bad}
        )
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "invalid_request"


def test_patch_capacity_is_returned_by_read_and_update(client):
    point = create_point(client)
    r = client.patch(f"/points/{point['id']}", json={"capacity": 5})
    assert r.status_code == 200, r.text
    assert r.json()["capacity"] == 5
    assert client.get(f"/points/{point['id']}").json()["capacity"] == 5


def test_patch_capacity_range_validation(client):
    point = create_point(client)
    for bad in (0, 21):
        r = client.patch(f"/points/{point['id']}", json={"capacity": bad})
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "invalid_request"
    assert client.get(f"/points/{point['id']}").json()["capacity"] == 1


# ---------------------------------------------------------------------------
# Booking against capacity
# ---------------------------------------------------------------------------


def test_capacity_allows_that_many_concurrent_reservations(client):
    point = open_point(client, capacity=3)
    pid = point["id"]
    keys = ["a", "b", "c"]
    for key in keys:
        r = reserve(client, pid, key, at(9), at(10))
        assert r.status_code == 201, r.text
    # The fourth concurrent visitor is refused with the existing overlap error.
    full = reserve(client, pid, "d", at(9), at(10))
    assert full.status_code == 409
    err = full.json()["error"]
    assert err["code"] == "reservation_overlap"
    assert len(err["details"]["conflicting_reservation_ids"]) == 3
    assert err["details"]["capacity"] == 3
    assert err["details"]["peak_concurrency"] == 3
    # A window that merely touches the block stays bookable.
    touching = reserve(client, pid, "touch", at(10), at(11))
    assert touching.status_code == 201, touching.text


def test_capacity_is_evaluated_at_every_instant_of_the_window(client):
    point = open_point(client, capacity=2)
    pid = point["id"]
    # Two pairs fill disjoint sub-windows at peak 2, with a free gap:
    # (a1,a2) on 09:00-09:30 and (b1,b2) on 10:00-10:30.
    reserve(client, pid, "a1", at(9), at(9, 30))
    reserve(client, pid, "a2", at(9), at(9, 30))
    reserve(client, pid, "b1", at(10), at(10, 30))
    reserve(client, pid, "b2", at(10), at(10, 30))
    # A window spanning both blocks has peak 2 == capacity somewhere -> full,
    # even though 09:30-10:00 in the middle is completely free.
    r = reserve(client, pid, "span", at(9, 15), at(10, 15))
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "reservation_overlap"
    assert r.json()["error"]["details"]["peak_concurrency"] == 2
    # A window contained in the free gap merely touches both blocks: free.
    ok = reserve(client, pid, "gap", at(9, 30), at(10))
    assert ok.status_code == 201, ok.text


def test_reschedule_moves_capacity_and_ignores_its_own_window(client):
    point = open_point(client, capacity=2)
    pid = point["id"]
    r1 = reserve(client, pid, "a", at(9), at(10)).json()
    # Two *other* visitors already fill the later window at capacity 2.
    reserve(client, pid, "b1", at(10, 30), at(11))
    reserve(client, pid, "b2", at(10, 30), at(11))
    # Moving r1 onto their window would make three concurrent visitors: the
    # move is refused (its own old window does not count against it).
    moved = client.post(
        f"/points/{pid}/reservations/{r1['id']}/reschedule",
        json={
            "cancel_token": r1["cancel_token"],
            "idempotency_key": "move",
            "expected_revision": 1,
            "start": iso(at(10, 30)),
            "end": iso(at(11)),
        },
    )
    assert moved.status_code == 409
    assert moved.json()["error"]["code"] == "reservation_overlap"
    # Moving just past that block (touching at 11:00) is free and succeeds.
    moved = client.post(
        f"/points/{pid}/reservations/{r1['id']}/reschedule",
        json={
            "cancel_token": r1["cancel_token"],
            "idempotency_key": "move",
            "expected_revision": 1,
            "start": iso(at(11)),
            "end": iso(at(11, 30)),
        },
    )
    assert moved.status_code == 200, moved.text
    assert moved.json()["revision"] == 2


def test_reassign_target_point_capacity_is_enforced(client):
    src = open_point(client, capacity=1)
    tgt = open_point(client, capacity=1)
    holder = reserve(client, tgt["id"], "h", at(9), at(10)).json()
    mine = reserve(client, src["id"], "m", at(9), at(10)).json()
    r = client.post(
        f"/points/{src['id']}/reservations/{mine['id']}/reassign",
        json={
            "cancel_token": mine["cancel_token"],
            "idempotency_key": "re",
            "expected_revision": 1,
            "target_point_id": tgt["id"],
            "start": iso(at(9)),
            "end": iso(at(10)),
        },
    )
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "reservation_overlap"
    assert r.json()["error"]["details"]["conflicting_reservation_ids"] == [
        holder["id"]
    ]
    # The source reservation is untouched by the rolled-back reassign.
    assert (
        client.post(
            f"/points/{src['id']}/reservations/{mine['id']}/cancel",
            json={"cancel_token": mine["cancel_token"]},
        ).status_code
        == 200
    )


# ---------------------------------------------------------------------------
# Capacity decrease
# ---------------------------------------------------------------------------


def test_capacity_decrease_below_future_concurrency_is_refused(client):
    point = open_point(client, capacity=3)
    pid = point["id"]
    for key in ("a", "b", "c"):
        assert reserve(client, pid, key, at(9), at(10)).status_code == 201
    r = client.patch(f"/points/{pid}", json={"capacity": 2})
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "capacity_conflict"
    assert err["details"]["requested_capacity"] == 2
    assert err["details"]["current_capacity"] == 3
    assert err["details"]["future_peak_concurrency"] == 3
    # The old value is unchanged.
    assert client.get(f"/points/{pid}").json()["capacity"] == 3


def test_capacity_decrease_is_allowed_when_bookings_still_fit(client):
    point = open_point(client, capacity=3)
    pid = point["id"]
    reserve(client, pid, "a", at(9), at(9, 30))
    # Two holders overlap by 15 minutes (09:15-09:30): peak 2 there.
    reserve(client, pid, "b", at(9, 15), at(10))
    r = client.patch(f"/points/{pid}", json={"capacity": 2})
    assert r.status_code == 200, r.text
    fetched = client.get(f"/points/{pid}").json()
    assert fetched["capacity"] == 2
    # Lowering to 1 is still refused: the 09:15-09:30 overlap peaks at 2.
    r = client.patch(f"/points/{pid}", json={"capacity": 1})
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "capacity_conflict"
    assert client.get(f"/points/{pid}").json()["capacity"] == 2


def test_capacity_decrease_equal_to_existing_peak_is_allowed(client):
    point = open_point(client, capacity=3)
    pid = point["id"]
    reserve(client, pid, "a", at(9), at(10))
    reserve(client, pid, "b", at(9), at(10))
    # Future peak is 2: lowering exactly to 2 is allowed (only strictly below
    # the peak is refused); lowering to 1 is not.
    ok = client.patch(f"/points/{pid}", json={"capacity": 2})
    assert ok.status_code == 200, ok.text
    assert client.get(f"/points/{pid}").json()["capacity"] == 2
    refused = client.patch(f"/points/{pid}", json={"capacity": 1})
    assert refused.status_code == 409
    assert refused.json()["error"]["code"] == "capacity_conflict"
    assert client.get(f"/points/{pid}").json()["capacity"] == 2


def test_cancelled_reservations_do_not_block_capacity_decrease(client):
    point = open_point(client, capacity=2)
    pid = point["id"]
    a = reserve(client, pid, "a", at(9), at(10)).json()
    b = reserve(client, pid, "b", at(9), at(10)).json()
    cancel(client, pid, a["id"], a["cancel_token"])
    cancel(client, pid, b["id"], b["cancel_token"])
    r = client.patch(f"/points/{pid}", json={"capacity": 1})
    assert r.status_code == 200, r.text
    assert client.get(f"/points/{pid}").json()["capacity"] == 1


def test_failed_capacity_decrease_changes_nothing_and_books_stand(client):
    point = open_point(client, capacity=2)
    pid = point["id"]
    reserve(client, pid, "a", at(9), at(10))
    reserve(client, pid, "b", at(9), at(10))
    refused = client.patch(f"/points/{pid}", json={"capacity": 1})
    assert refused.status_code == 409
    assert refused.json()["error"]["details"]["future_peak_concurrency"] == 2
    # Capacity 2 still holds: a third concurrent booking is still refused and
    # the stored value is unchanged.
    assert reserve(client, pid, "c", at(9), at(10)).status_code == 409
    assert client.get(f"/points/{pid}").json()["capacity"] == 2
    # A window touching the block stays bookable.
    assert reserve(client, pid, "after", at(10), at(10, 30)).status_code == 201


# ---------------------------------------------------------------------------
# Waitlist admission under capacity
# ---------------------------------------------------------------------------


def test_waitlist_only_for_full_capacity_free_window_uses_existing_error(client):
    point = open_point(client, capacity=2)
    pid = point["id"]
    reserve(client, pid, "a", at(9), at(10))
    # One free slot left: waitlist must not apply (existing free error).
    r = waitlist(client, pid, "w-free", at(9), at(10))
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "waitlist_window_available"
    # Fill the second slot, then the same window is a valid waitlist target.
    reserve(client, pid, "b", at(9), at(10))
    r = waitlist(client, pid, "w-full", at(9), at(10))
    assert r.status_code == 201, r.text
    assert r.json()["status"] == "waiting"


def test_waitlist_accepts_window_full_only_on_part_of_its_span(client):
    point = open_point(client, capacity=1)
    pid = point["id"]
    # Holder 09:00-09:30.  The requested span is peak 1 at 09:15 but free
    # from 09:30 on: capacity semantics still admit it to the waitlist.
    reserve(client, pid, "a", at(9), at(9, 30))
    r = waitlist(client, pid, "span", at(9, 15), at(9, 45))
    assert r.status_code == 201, r.text
    # A strictly touching window has room everywhere: not waitlist material.
    r = waitlist(client, pid, "touch", at(9, 30), at(10))
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "waitlist_window_available"


# ---------------------------------------------------------------------------
# Promotions after release / capacity increase
# ---------------------------------------------------------------------------


def test_cancel_promotes_every_entry_that_now_fits_in_order(client):
    point = open_point(client, capacity=3)
    pid = point["id"]
    holders = []
    for key in ("h1", "h2", "h3"):
        holders.append(reserve(client, pid, key, at(9), at(10)).json())
    w1 = waitlist(client, pid, "w1", at(9), at(10)).json()
    w2 = waitlist(client, pid, "w2", at(9), at(10)).json()
    # Cancel two holders: both waiters must now be promoted, in apply order.
    cancel(client, pid, holders[0]["id"], holders[0]["cancel_token"])
    cancel(client, pid, holders[1]["id"], holders[1]["cancel_token"])
    s1 = status(client, pid, w1["id"], w1["query_token"]).json()
    s2 = status(client, pid, w2["id"], w2["query_token"]).json()
    assert s1["status"] == "promoted"
    assert s2["status"] == "promoted"
    # The point is exactly full again; one holder remains plus two promotions.
    full = reserve(client, pid, "extra", at(9), at(10))
    assert full.status_code == 409


def test_single_release_promotes_multiple_disjoint_entries_in_one_sweep(client):
    # Capacity 1 (default): one holder spans 09:00-11:00.  Two applicants
    # queue for the two touching halves 09:00-10:00 and 10:00-11:00.  One
    # cancellation runs one sweep; both entries now fit simultaneously (they
    # merely touch at 10:00), so the single sweep promotes both, in order.
    point = open_point(client, capacity=1)
    pid = point["id"]
    holder = reserve(client, pid, "h", at(9), at(11)).json()
    w1 = waitlist(client, pid, "w1", at(9), at(10)).json()
    w2 = waitlist(client, pid, "w2", at(10), at(11)).json()
    cancel(client, pid, holder["id"], holder["cancel_token"])
    s1 = status(client, pid, w1["id"], w1["query_token"]).json()
    s2 = status(client, pid, w2["id"], w2["query_token"]).json()
    assert s1["status"] == "promoted", s1
    assert s2["status"] == "promoted", s2
    # The two promoted halves refill the whole block end-to-end.
    assert reserve(client, pid, "full", at(9, 30), at(10, 30)).status_code == 409
    assert reserve(client, pid, "before", at(8, 30), at(9)).status_code == 201
    assert reserve(client, pid, "after", at(11), at(11, 30)).status_code == 201


def test_promotion_skips_entries_that_do_not_fit_yet_keeps_order(client):
    point = open_point(client, capacity=3)
    pid = point["id"]
    # Six active bookings: three on the 09:00-09:30 half and three on the
    # 09:30-10:00 half (they merely touch at 09:30, so each half peaks at 3).
    first_half = [
        reserve(client, pid, f"f{i}", at(9), at(9, 30)).json() for i in range(3)
    ]
    for i in range(3):
        assert reserve(client, pid, f"s{i}", at(9, 30), at(10)).status_code == 201
    # w1 spans the whole hour: releasing one first-half booking still leaves
    # peak 3 on the second half, so w1 cannot promote and keeps waiting.
    w1 = waitlist(client, pid, "w1", at(9), at(10)).json()
    # w2 only wants 09:00-09:30: after one first-half release its window
    # peaks at 2 < capacity 3, so it promotes.
    w2 = waitlist(client, pid, "w2", at(9), at(9, 30)).json()
    cancel(client, pid, first_half[0]["id"], first_half[0]["cancel_token"])
    waiting = status(client, pid, w1["id"], w1["query_token"]).json()
    assert waiting["status"] == "waiting"
    assert waiting["position"] == 1
    promoted = status(client, pid, w2["id"], w2["query_token"]).json()
    assert promoted["status"] == "promoted"
    assert promoted["reservation"]["start"] == iso(at(9))
    # The first half is full again (two originals + the promoted w2) and the
    # untouched second half is still full on its own.
    assert reserve(client, pid, "full1", at(9), at(9, 30)).status_code == 409
    assert reserve(client, pid, "full2", at(9, 30), at(10)).status_code == 409


def test_capacity_increase_promotes_waiters_in_same_transaction(client):
    point = open_point(client, capacity=1)
    pid = point["id"]
    reserve(client, pid, "h", at(9), at(10))
    w1 = waitlist(client, pid, "w1", at(9), at(10)).json()
    w2 = waitlist(client, pid, "w2", at(9), at(10)).json()
    # Raise to 4: three slots open; the two applicants promote at once and
    # two free slots remain.
    r = client.patch(f"/points/{pid}", json={"capacity": 4})
    assert r.status_code == 200, r.text
    assert r.json()["capacity"] == 4
    s1 = status(client, pid, w1["id"], w1["query_token"]).json()
    s2 = status(client, pid, w2["id"], w2["query_token"]).json()
    assert s1["status"] == "promoted"
    assert s2["status"] == "promoted"
    assert reserve(client, pid, "third", at(9), at(10)).status_code == 201
    assert reserve(client, pid, "fourth", at(9), at(10)).status_code == 409


def test_capacity_increase_promotes_only_what_fits(client):
    point = open_point(client, capacity=1)
    pid = point["id"]
    reserve(client, pid, "h", at(9), at(10))
    w1 = waitlist(client, pid, "w1", at(9), at(10)).json()
    w2 = waitlist(client, pid, "w2", at(9), at(10)).json()
    r = client.patch(f"/points/{pid}", json={"capacity": 2})
    assert r.status_code == 200
    first = status(client, pid, w1["id"], w1["query_token"]).json()
    second = status(client, pid, w2["id"], w2["query_token"]).json()
    assert first["status"] == "promoted"
    assert second["status"] == "waiting"
    assert second["position"] == 1
    # A subsequent release promotes the remaining applicant.
    promoted_id = first["reservation"]["id"]
    promoted_token = first["reservation"]["cancel_token"]
    cancel(client, pid, promoted_id, promoted_token)
    second = status(client, pid, w2["id"], w2["query_token"]).json()
    assert second["status"] == "promoted"


def test_capacity_increase_without_rules_does_not_sweep_and_works(client):
    point = create_point(client, capacity=1)
    # No published calendar: increasing capacity simply succeeds.
    r = client.patch(f"/points/{point['id']}", json={"capacity": 4})
    assert r.status_code == 200
    assert r.json()["capacity"] == 4


# ---------------------------------------------------------------------------
# Trip / reassign use the same capacity
# ---------------------------------------------------------------------------


def test_trip_stop_respects_point_capacity(client):
    point = open_point(client, capacity=1)
    pid = point["id"]
    # A second point with an open window later in the day is needed for a
    # valid 2-stop trip; it has capacity 1 and is free.
    other = open_point(client, capacity=1)
    reserve(client, pid, "h", at(9), at(9, 30))
    r = client.post(
        "/trips",
        json={
            "idempotency_key": "trip-1",
            "stops": [
                {"point_id": pid, "start": iso(at(9)), "end": iso(at(9, 30))},
                {
                    "point_id": other["id"],
                    "start": iso(at(10)),
                    "end": iso(at(10, 30)),
                },
            ],
        },
    )
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "reservation_overlap"
    assert err["details"]["stop_index"] == 0
    assert err["details"]["point_id"] == pid
    # Nothing was booked at either stop (whole-group atomicity).
    assert reserve(client, other["id"], "free-check", at(10), at(10, 30)).status_code == 201


def test_trip_stop_fits_when_capacity_has_room(client):
    p1 = open_point(client, capacity=2)
    p2 = open_point(client, capacity=2)
    reserve(client, p1["id"], "h1", at(9), at(9, 30))
    reserve(client, p2["id"], "h2", at(10), at(10, 30))
    r = client.post(
        "/trips",
        json={
            "idempotency_key": "trip-ok",
            "stops": [
                {"point_id": p1["id"], "start": iso(at(9)), "end": iso(at(9, 30))},
                {"point_id": p2["id"], "start": iso(at(10)), "end": iso(at(10, 30))},
            ],
        },
    )
    assert r.status_code == 201, r.text
    stops = r.json()["stops"]
    assert len(stops) == 2
    # Each stop now holds exactly 2 concurrent reservations: full.
    assert reserve(client, p1["id"], "x1", at(9), at(9, 30)).status_code == 409
    assert reserve(client, p2["id"], "x2", at(10), at(10, 30)).status_code == 409


# ---------------------------------------------------------------------------
# Concurrency: never oversubscribe
# ---------------------------------------------------------------------------


def test_concurrent_bookings_never_exceed_capacity(client):
    point = open_point(client, capacity=3)
    pid = point["id"]
    threads_n = 8
    barrier = threading.Barrier(threads_n)
    outcomes = []

    def attempt(i):
        barrier.wait()
        r = reserve(client, pid, f"race-{i}", at(9), at(10))
        outcomes.append(r.status_code)

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(threads_n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(outcomes).count(201) == 3
    assert sorted(outcomes).count(409) == 5
    # Database-wide invariant check via the public API: a fourth concurrent
    # reservation is impossible.
    assert reserve(client, pid, "after", at(9), at(10)).status_code == 409


def test_concurrent_capacity_increase_and_cancel_promotions_are_safe(client):
    point = open_point(client, capacity=1)
    pid = point["id"]
    holder = reserve(client, pid, "h", at(9), at(10)).json()
    waiters = [waitlist(client, pid, f"w{i}", at(9), at(10)).json() for i in range(3)]
    barrier = threading.Barrier(2)
    statuses = {}

    def do_cancel():
        barrier.wait()
        statuses["cancel"] = cancel(
            client, pid, holder["id"], holder["cancel_token"]
        ).status_code

    def do_increase():
        barrier.wait()
        r = client.patch(f"/points/{pid}", json={"capacity": 3})
        statuses["patch"] = r.status_code

    threads = [threading.Thread(target=do_cancel), threading.Thread(target=do_increase)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert statuses == {"cancel": 200, "patch": 200}
    states = [
        status(client, pid, w["id"], w["query_token"]).json()["status"]
        for w in waiters
    ]
    # The holder is gone and capacity is 3: exactly three slots, all promoted,
    # regardless of which transaction committed first.
    assert states == ["promoted", "promoted", "promoted"]


def test_capacity_increase_sweep_failure_rolls_back_the_capacity_change(
    client, monkeypatch
):
    import pytest

    import app.db as db_module
    import app.scheduler as scheduler_module

    point = open_point(client, capacity=1)
    pid = point["id"]
    reserve(client, pid, "h", at(9), at(10))
    w = waitlist(client, pid, "w", at(9), at(10)).json()
    real_build = scheduler_module.build_open_intervals

    def boom(*args, **kwargs):
        raise RuntimeError("capacity sweep failure simulated")

    monkeypatch.setattr(db_module, "build_open_intervals", boom)
    try:
        with pytest.raises(RuntimeError, match="capacity sweep failure simulated"):
            client.patch(f"/points/{pid}", json={"capacity": 3})
    finally:
        monkeypatch.setattr(db_module, "build_open_intervals", real_build)

    # The capacity change rolled back together with the failed sweep.
    assert client.get(f"/points/{pid}").json()["capacity"] == 1
    body = status(client, pid, w["id"], w["query_token"]).json()
    assert body["status"] == "waiting"
    # Retrying now succeeds and promotes the applicant.
    ok = client.patch(f"/points/{pid}", json={"capacity": 3})
    assert ok.status_code == 200
    assert (
        status(client, pid, w["id"], w["query_token"]).json()["status"]
        == "promoted"
    )


def test_database_never_oversubscribed_after_concurrent_booking(client):
    from app import db

    point = open_point(client, capacity=4)
    pid = point["id"]
    threads_n = 12
    barrier = threading.Barrier(threads_n)
    outcomes = []

    def attempt(i):
        barrier.wait()
        outcomes.append(
            reserve(client, pid, f"c-{i}", at(9, (i % 4) * 5), at(10)).status_code
        )

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(threads_n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert outcomes.count(201) <= 4
    # Invariant straight from the database: at the busiest instant, the count
    # of active reservations never exceeds the point's capacity.
    with db.conn() as connection:
        row = connection.cursor().execute(
            "SELECT COALESCE(MAX(c), 0) AS peak FROM ("
            "  SELECT s.t, (SELECT COUNT(*) FROM reservations r "
            "    WHERE r.point_id = p.id AND r.status = 'active' "
            "      AND r.start_utc <= s.t AND r.end_utc > s.t) AS c "
            "  FROM points p, LATERAL ("
            "    SELECT start_utc AS t FROM reservations "
            "    WHERE point_id = p.id AND status = 'active') s "
            "  WHERE p.id = %s"
            ") x",
            (pid,),
        ).fetchone()
    assert row["peak"] <= 4
    assert outcomes.count(201) == row["peak"]
