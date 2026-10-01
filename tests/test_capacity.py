"""End-to-end tests for per-point concurrent-visitor capacity (容量).

Covers:
* capacity on create defaults to 1 and is returned by read/patch;
* 1..20 range validation;
* multiple concurrent bookings up to capacity, half-open [start, end);
* reschedule / reassign / trip share the same capacity rule;
* lowering capacity is refused against future concurrency (old value kept);
* raising capacity (and cancelling / rescheduling) promotes waitlist entries
  in application order, as many as fit, in the same transaction;
* waitlist only accepts calendar-valid windows that are full;
* concurrent writes never overbook.
"""

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


def create_point(client, capacity=None, tz="UTC", name="容量测试点"):
    body = {"name": name, "timezone": tz}
    if capacity is not None:
        body["capacity"] = capacity
    r = client.post("/points", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def open_point(client, capacity=2, weekly=None, tz="UTC"):
    point = create_point(client, capacity=capacity, tz=tz)
    pid = point["id"]
    r = client.post(
        f"/points/{pid}/rules",
        json={
            "expected_version": 0,
            "rules": {
                "weekly": weekly
                if weekly is not None
                else [
                    {
                        "weekday": at(0).isoweekday(),
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
    return pid


def reserve(client, pid, key, start, end):
    return client.post(
        f"/points/{pid}/reservations",
        json={"idempotency_key": key, "start": iso(start), "end": iso(end)},
    )


def cancel(client, pid, rid, token):
    return client.post(
        f"/points/{pid}/reservations/{rid}/cancel",
        json={"cancel_token": token},
    )


def reschedule(client, pid, rid, token, key, start, end, revision):
    return client.post(
        f"/points/{pid}/reservations/{rid}/reschedule",
        json={
            "cancel_token": token,
            "idempotency_key": key,
            "expected_revision": revision,
            "start": iso(start),
            "end": iso(end),
        },
    )


def waitlist(client, pid, key, start, end):
    return client.post(
        f"/points/{pid}/waitlist",
        json={"idempotency_key": key, "start": iso(start), "end": iso(end)},
    )


def wstatus(client, pid, wid, token):
    return client.get(
        f"/points/{pid}/waitlist/{wid}", params={"query_token": token}
    )


# ---------------------------------------------------------------------------
# Capacity on points
# ---------------------------------------------------------------------------


def test_capacity_defaults_to_one_and_is_returned(client):
    point = create_point(client)
    assert point["capacity"] == 1
    fetched = client.get(f"/points/{point['id']}").json()
    assert fetched["capacity"] == 1


def test_capacity_is_set_at_creation_and_patched(client):
    point = create_point(client, capacity=5)
    assert point["capacity"] == 5
    pid = point["id"]
    r = client.patch(f"/points/{pid}", json={"capacity": 8})
    assert r.status_code == 200
    assert r.json()["capacity"] == 8
    assert client.get(f"/points/{pid}").json()["capacity"] == 8
    # A name-only patch leaves capacity untouched.
    r = client.patch(f"/points/{pid}", json={"name": "改名"})
    assert r.status_code == 200
    assert r.json()["capacity"] == 8
    assert r.json()["name"] == "改名"


def test_capacity_bounds_are_rejected_as_invalid_request(client):
    for bad in (0, -1, 21, 100):
        r = client.post(
            "/points", json={"name": "越界", "timezone": "UTC", "capacity": bad}
        )
        assert r.status_code == 422, bad
        assert r.json()["error"]["code"] == "invalid_request"
    point = create_point(client, capacity=1)
    r = client.patch(f"/points/{point['id']}", json={"capacity": 0})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"
    r = client.patch(f"/points/{point['id']}", json={"capacity": 21})
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Booking under capacity, half-open [start, end)
# ---------------------------------------------------------------------------


def test_concurrent_bookings_up_to_capacity_then_full(client):
    pid = open_point(client, capacity=3)
    r1 = reserve(client, pid, "a", at(9), at(10))
    r2 = reserve(client, pid, "b", at(9, 15), at(10, 15))
    r3 = reserve(client, pid, "c", at(9, 30), at(10, 30))
    assert [r.status_code for r in (r1, r2, r3)] == [201, 201, 201]
    # A fourth strictly overlapping window is full.
    full = reserve(client, pid, "d", at(9, 45), at(10, 45))
    assert full.status_code == 409
    err = full.json()["error"]
    assert err["code"] == "reservation_overlap"
    assert err["details"]["capacity"] == 3
    assert err["details"]["concurrent_reservations"] == 3
    assert set(err["details"]["conflicting_reservation_ids"]) == {
        r1.json()["id"],
        r2.json()["id"],
        r3.json()["id"],
    }


def test_non_overlapping_windows_do_not_count_against_capacity(client):
    pid = open_point(client, capacity=1)
    assert reserve(client, pid, "a", at(9), at(10)).status_code == 201
    # Back-to-back: half-open intervals touch, so the second is allowed.
    assert reserve(client, pid, "b", at(10), at(11)).status_code == 201
    # Strictly inside the first would be a second concurrent visitor: refused.
    r = reserve(client, pid, "c", at(9, 30), at(10, 30))
    assert r.status_code == 409
    assert r.json()["error"]["details"]["capacity"] == 1


def test_capacity_is_per_instant_partial_overlap_fits(client):
    # capacity 2: at no single instant do three bookings overlap.
    pid = open_point(client, capacity=2)
    assert reserve(client, pid, "a", at(9), at(10)).status_code == 201
    assert reserve(client, pid, "b", at(10), at(11)).status_code == 201
    # Overlaps both windows only by touching (10:00) -> still free.
    assert reserve(client, pid, "c", at(10), at(10, 30)).status_code == 201
    # Now 09:30-10:00 overlaps a; 10:00-10:30 overlaps b,c (2 already) ->
    # the window needs a third seat at 10:00..10:30, so it is refused.
    r = reserve(client, pid, "d", at(9, 30), at(10, 30))
    assert r.status_code == 409
    assert r.json()["error"]["details"]["concurrent_reservations"] == 2


def test_cancelled_reservation_releases_a_seat(client):
    pid = open_point(client, capacity=2)
    a = reserve(client, pid, "a", at(9), at(10)).json()
    b = reserve(client, pid, "b", at(9), at(10)).json()
    assert reserve(client, pid, "c", at(9), at(10)).status_code == 409
    cancel(client, pid, a["id"], a["cancel_token"])
    assert reserve(client, pid, "c", at(9), at(10)).status_code == 201
    # Full again (b + c).
    assert reserve(client, pid, "d", at(9), at(10)).status_code == 409


# ---------------------------------------------------------------------------
# Reschedule / reassign / trips
# ---------------------------------------------------------------------------


def test_reschedule_uses_capacity_ignoring_its_own_old_window(client):
    pid = open_point(client, capacity=2)
    me = reserve(client, pid, "me", at(9), at(10)).json()
    reserve(client, pid, "other", at(10), at(11))
    # Moving onto other's window: only the two of them at 10:00..10:30 -> fits.
    r = reschedule(
        client, pid, me["id"], me["cancel_token"], "move", at(10), at(10, 30), 1
    )
    assert r.status_code == 200, r.text
    # A third booking overlapping them is full.
    assert reserve(client, pid, "third", at(10), at(10, 30)).status_code == 409


def test_reassign_target_uses_target_capacity(client):
    src = open_point(client, capacity=1)
    tgt = open_point(client, capacity=2)
    t1 = reserve(client, tgt, "t1", at(10), at(11)).json()
    me = reserve(client, src, "me", at(9), at(10)).json()
    body = {
        "cancel_token": me["cancel_token"],
        "idempotency_key": "reassign-1",
        "expected_revision": 1,
        "target_point_id": tgt,
        "start": iso(at(10, 30)),
        "end": iso(at(11, 30)),
    }
    r = client.post(
        f"/points/{src}/reservations/{me['id']}/reassign", json=body
    )
    assert r.status_code == 200, r.text
    # Target now has t1 and the reassigned booking overlapping at 10:30-11:00.
    full = reserve(client, tgt, "t3", at(10, 30), at(11))
    assert full.status_code == 409
    assert full.json()["error"]["details"]["capacity"] == 2


def test_reassign_refused_when_target_full_leaves_both_points(client):
    src = open_point(client, capacity=1)
    tgt = open_point(client, capacity=1)
    blocker = reserve(client, tgt, "blocker", at(10), at(11)).json()
    me = reserve(client, src, "me", at(9), at(10)).json()
    r = client.post(
        f"/points/{src}/reservations/{me['id']}/reassign",
        json={
            "cancel_token": me["cancel_token"],
            "idempotency_key": "reassign-full",
            "expected_revision": 1,
            "target_point_id": tgt,
            "start": iso(at(10, 30)),
            "end": iso(at(11, 30)),
        },
    )
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "reservation_overlap"
    # Source untouched, target occupancy unchanged.
    assert (
        client.post(
            f"/points/{src}/reservations/{me['id']}/cancel",
            json={"cancel_token": me["cancel_token"]},
        ).status_code
        == 200
    )
    assert blocker["id"]


def test_trip_stops_are_checked_against_each_point_capacity(client):
    p1 = open_point(client, capacity=1)
    p2 = open_point(client, capacity=1)
    # Fill p2's stop window.
    reserve(client, p2, "p2-holder", at(10), at(10, 30))
    r = client.post(
        "/trips",
        json={
            "idempotency_key": "cap-trip-1",
            "stops": [
                {"point_id": p1, "start": iso(at(9)), "end": iso(at(9, 30))},
                {"point_id": p2, "start": iso(at(10)), "end": iso(at(10, 30))},
            ],
        },
    )
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "reservation_overlap"
    assert err["details"]["stop_index"] == 1
    assert err["details"]["point_id"] == p2
    assert err["details"]["capacity"] == 1
    # First stop was not booked either.
    assert reserve(client, p1, "proof", at(9), at(9, 30)).status_code == 201


def test_trip_succeeds_when_every_stop_has_a_free_seat(client):
    p1 = open_point(client, capacity=2)
    p2 = open_point(client, capacity=2)
    reserve(client, p1, "p1-other", at(9), at(9, 30))
    reserve(client, p2, "p2-other", at(10), at(10, 30))
    r = client.post(
        "/trips",
        json={
            "idempotency_key": "cap-trip-2",
            "stops": [
                {"point_id": p1, "start": iso(at(9)), "end": iso(at(9, 30))},
                {"point_id": p2, "start": iso(at(10)), "end": iso(at(10, 30))},
            ],
        },
    )
    assert r.status_code == 201, r.text
    # Both points are now at capacity.
    assert reserve(client, p1, "full1", at(9), at(9, 30)).status_code == 409
    assert reserve(client, p2, "full2", at(10), at(10, 30)).status_code == 409


# ---------------------------------------------------------------------------
# Lowering capacity
# ---------------------------------------------------------------------------


def test_lowering_capacity_above_future_concurrency_succeeds(client):
    pid = open_point(client, capacity=5)
    a = reserve(client, pid, "a", at(9), at(10)).json()
    reserve(client, pid, "b", at(10), at(11))  # touches a: peak only 1
    # Peak future concurrency is 1, so lowering to 1 is allowed.
    r = client.patch(f"/points/{pid}", json={"capacity": 1})
    assert r.status_code == 200
    assert r.json()["capacity"] == 1
    assert a["id"]


def test_lowering_capacity_below_future_concurrency_is_refused(client):
    pid = open_point(client, capacity=3)
    a = reserve(client, pid, "a", at(9), at(10)).json()
    b = reserve(client, pid, "b", at(9, 30), at(10, 30)).json()
    before = client.get(f"/points/{pid}").json()
    r = client.patch(f"/points/{pid}", json={"capacity": 1})
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "capacity_conflict"
    assert err["details"]["requested_capacity"] == 1
    assert err["details"]["max_concurrent_reservations"] == 2
    assert set(err["details"]["conflicting_reservation_ids"]) == {
        a["id"],
        b["id"],
    }
    # Old value is unchanged.
    after = client.get(f"/points/{pid}").json()
    assert after["capacity"] == 3
    assert after["updated_at"] == before["updated_at"]
    # The exact peak is the floor: lowering to 2 works.
    r = client.patch(f"/points/{pid}", json={"capacity": 2})
    assert r.status_code == 200
    assert r.json()["capacity"] == 2


def test_lowering_capacity_ignores_past_and_cancelled_reservations(client):
    pid = open_point(client, capacity=3)
    a = reserve(client, pid, "a", at(9), at(10)).json()
    b = reserve(client, pid, "b", at(9), at(10)).json()
    cancel(client, pid, b["id"], b["cancel_token"])
    # Only one future active booking now: lowering to 1 is fine.
    r = client.patch(f"/points/{pid}", json={"capacity": 1})
    assert r.status_code == 200
    assert r.json()["capacity"] == 1
    assert a["id"]


# ---------------------------------------------------------------------------
# Waitlist vs. capacity
# ---------------------------------------------------------------------------


def test_waitlist_accepted_when_full_rejected_when_seats_remain(client):
    pid = open_point(client, capacity=2)
    reserve(client, pid, "a", at(9), at(10))
    # One seat remains: waitlist must not accept.
    r = waitlist(client, pid, "early", at(9), at(10))
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "waitlist_window_available"
    assert r.json()["error"]["details"]["capacity"] == 2
    assert r.json()["error"]["details"]["concurrent_reservations"] == 1
    reserve(client, pid, "b", at(9), at(10))
    # Full now: the waitlist accepts.
    r = waitlist(client, pid, "w", at(9), at(10))
    assert r.status_code == 201, r.text
    assert r.json()["status"] == "waiting"


def test_touching_window_is_not_full_and_not_waitlistable(client):
    pid = open_point(client, capacity=1)
    reserve(client, pid, "a", at(9), at(10))
    # [10:00, 11:00) only touches the holder: seat available.
    r = waitlist(client, pid, "touch", at(10), at(11))
    assert r.status_code == 409
    assert r.json()["error"]["code"] == "waitlist_window_available"
    assert reserve(client, pid, "direct", at(10), at(11)).status_code == 201


def test_cancel_promotes_as_many_waiting_entries_as_fit(client):
    pid = open_point(client, capacity=2)
    # Two holders over 09:00-10:00 fill the point.
    h1 = reserve(client, pid, "h1", at(9), at(10)).json()
    h2 = reserve(client, pid, "h2", at(9), at(10)).json()
    w1 = waitlist(client, pid, "w1", at(9), at(10)).json()
    w2 = waitlist(client, pid, "w2", at(9), at(10)).json()
    w3 = waitlist(client, pid, "w3", at(9), at(10)).json()
    # Cancel both holders in the same request sequence; the first cancellation
    # frees one seat and promotes exactly one (capacity 2, one holder remains).
    cancel(client, pid, h1["id"], h1["cancel_token"])
    s1 = wstatus(client, pid, w1["id"], w1["query_token"]).json()
    s2 = wstatus(client, pid, w2["id"], w2["query_token"]).json()
    assert s1["status"] == "promoted"
    assert s2["status"] == "waiting"
    # The second release frees the other seat: w2 promotes, w3 stays queued
    # (promoted w1 plus promoted w2 fill the point again).
    cancel(client, pid, h2["id"], h2["cancel_token"])
    s2 = wstatus(client, pid, w2["id"], w2["query_token"]).json()
    s3 = wstatus(client, pid, w3["id"], w3["query_token"]).json()
    assert s2["status"] == "promoted"
    assert s3["status"] == "waiting"
    assert s3["position"] == 1
    # The point holds exactly two active reservations at 09:00-10:00.
    assert reserve(client, pid, "overflow", at(9), at(10)).status_code == 409


def test_raising_capacity_promotes_waitlist_in_application_order(client):
    pid = open_point(client, capacity=1)
    reserve(client, pid, "holder", at(9), at(10))
    entries = []
    for k in ("w1", "w2", "w3"):
        w = waitlist(client, pid, k, at(9), at(10)).json()
        entries.append(w)
    # Raising to 3: holder + two waiters fit; the first two applicants promote,
    # the third stays queued — all in the same transaction as the PATCH.
    r = client.patch(f"/points/{pid}", json={"capacity": 3})
    assert r.status_code == 200, r.text
    statuses = [
        wstatus(client, pid, w["id"], w["query_token"]).json()["status"]
        for w in entries
    ]
    assert statuses == ["promoted", "promoted", "waiting"]
    # Three active reservations at 09:00 now; a fourth is refused.
    assert reserve(client, pid, "fourth", at(9), at(10)).status_code == 409
    # Raising again to 4 promotes the remaining applicant in that transaction.
    r = client.patch(f"/points/{pid}", json={"capacity": 4})
    assert r.status_code == 200
    last = wstatus(
        client, pid, entries[2]["id"], entries[2]["query_token"]
    ).json()
    assert last["status"] == "promoted"


def test_capacity_increase_skips_entries_that_do_not_fit_then_promotes_later(
    client,
):
    pid = open_point(client, capacity=1)
    # Holder covers 09:00-10:00.  w1 wants a wider window (would overlap both
    # active bookings later); w2 wants exactly the freed shape.
    reserve(client, pid, "holder", at(9), at(10))
    reserve(client, pid, "neighbor", at(10), at(11))
    w1 = waitlist(client, pid, "w1", at(9, 30), at(10, 30)).json()
    w2 = waitlist(client, pid, "w2", at(9), at(10)).json()
    # Capacity 2: w1 needs two seats at 10:00..10:30 (neighbor + itself) — fits
    # by itself, but w2 must be judged in order: w1 first -> promoted; then w2
    # would overlap promoted w1 + neighbor at 10:00..10:30 (two), so with
    # capacity 2 it does not fit and stays queued.
    client.patch(f"/points/{pid}", json={"capacity": 2})
    s1 = wstatus(client, pid, w1["id"], w1["query_token"]).json()
    s2 = wstatus(client, pid, w2["id"], w2["query_token"]).json()
    assert s1["status"] == "promoted"
    assert s2["status"] == "waiting"
    # One more seat: w2 now fits and promotes in the PATCH transaction.
    client.patch(f"/points/{pid}", json={"capacity": 3})
    s2 = wstatus(client, pid, w2["id"], w2["query_token"]).json()
    assert s2["status"] == "promoted"


def test_promotion_failure_rolls_back_capacity_increase(client, monkeypatch):
    import pytest

    import app.db as db_module
    import app.scheduler as scheduler_module

    pid = open_point(client, capacity=1)
    reserve(client, pid, "holder", at(9), at(10))
    w = waitlist(client, pid, "w", at(9), at(10)).json()
    real_build = scheduler_module.build_open_intervals

    def boom(*args, **kwargs):
        raise RuntimeError("capacity sweep failure simulated")

    monkeypatch.setattr(db_module, "build_open_intervals", boom)
    try:
        with pytest.raises(RuntimeError, match="capacity sweep failure"):
            client.patch(f"/points/{pid}", json={"capacity": 4})
    finally:
        monkeypatch.setattr(db_module, "build_open_intervals", real_build)
    # The capacity change and the promotion both rolled back.
    assert client.get(f"/points/{pid}").json()["capacity"] == 1
    body = wstatus(client, pid, w["id"], w["query_token"]).json()
    assert body["status"] == "waiting"


# ---------------------------------------------------------------------------
# Concurrency: never overbook
# ---------------------------------------------------------------------------


def test_concurrent_bookings_never_exceed_capacity(client):
    pid = open_point(client, capacity=3)

    def book(key):
        return reserve(client, pid, key, at(9), at(10)).status_code

    barrier = threading.Barrier(6)
    results: list[int] = []

    def attempt(key):
        barrier.wait()
        results.append(book(key))

    threads = [
        threading.Thread(target=attempt, args=(f"race-{i}",)) for i in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results).count(201) == 3
    assert sorted(results).count(409) == 3


def test_concurrent_capacity_raise_and_cancel_promote_without_oversell(client):
    pid = open_point(client, capacity=1)
    holder = reserve(client, pid, "holder", at(9), at(10)).json()
    w1 = waitlist(client, pid, "w1", at(9), at(10)).json()
    w2 = waitlist(client, pid, "w2", at(9), at(10)).json()
    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def do_cancel():
        barrier.wait()
        cancel(client, pid, holder["id"], holder["cancel_token"])
        outcomes.append("cancel")

    def do_raise():
        barrier.wait()
        client.patch(f"/points/{pid}", json={"capacity": 2})
        outcomes.append("raise")

    threads = [threading.Thread(target=do_cancel), threading.Thread(target=do_raise)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    s1 = wstatus(client, pid, w1["id"], w1["query_token"]).json()
    s2 = wstatus(client, pid, w2["id"], w2["query_token"]).json()
    # Whichever order the two serialized transactions took, exactly two
    # reservations end up active at 09:00-10:00 and both waiters are promoted.
    assert s1["status"] == "promoted"
    assert s2["status"] == "promoted"
    overflow = reserve(client, pid, "overflow", at(9), at(10))
    assert overflow.status_code == 409
    assert overflow.json()["error"]["details"]["capacity"] == 2
