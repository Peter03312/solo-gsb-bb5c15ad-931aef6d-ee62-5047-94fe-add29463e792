"""End-to-end tests for atomic multi-stop trip bookings (跨回收点行程整体预约)."""

from __future__ import annotations

import itertools
import threading
from datetime import datetime, timedelta, timezone

UTC = timezone.utc

# Trip idempotency keys are one global scope, so every test uses fresh keys.
_keys = itertools.count(1)


def unique_key(prefix="trip"):
    return f"{prefix}-{next(_keys)}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def create_point(client, tz="UTC"):
    r = client.post("/points", json={"name": "行程测试点", "timezone": tz})
    assert r.status_code == 201, r.text
    return r.json()["id"]


def open_point(client, weekly=None, tz="UTC"):
    """Point with a weekly rule; default 08:00-12:00 tomorrow (UTC)."""
    pid = create_point(client, tz=tz)
    tomorrow = (datetime.now(UTC) + timedelta(days=1)).date()
    r = client.post(
        f"/points/{pid}/rules",
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
    return pid


def at(hour, minute=0, day_offset=1):
    """Fixed UTC wall-clock ``day_offset`` days from today."""
    day = (datetime.now(UTC) + timedelta(days=day_offset)).date()
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=UTC)


def iso(dt):
    return dt.isoformat().replace("+00:00", "Z")


def stop(point_id, start, end):
    return {"point_id": point_id, "start": iso(start), "end": iso(end)}


def book_trip(client, key, stops):
    return client.post(
        "/trips", json={"idempotency_key": key, "stops": stops}
    )


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


def waitlist(client, pid, key, start, end):
    return client.post(
        f"/points/{pid}/waitlist",
        json={"idempotency_key": key, "start": iso(start), "end": iso(end)},
    )


def open_points(client, count):
    return [open_point(client) for _ in range(count)]


def two_stop_trip(client, key=None):
    """A valid two-stop trip at two fresh points: 09:00-09:30, 10:00-10:30."""
    key = key or unique_key()
    p1, p2 = open_points(client, 2)
    stops = [stop(p1, at(9), at(9, 30)), stop(p2, at(10), at(10, 30))]
    return (p1, p2), stops, book_trip(client, key, stops)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_trip_success_books_all_stops_in_input_order(client):
    (p1, p2), stops, r = two_stop_trip(client, key="trip-success")
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["idempotency_key"] == "trip-success"
    assert body["trip_id"] > 0
    assert body["created_at"]
    assert [s["point_id"] for s in body["stops"]] == [p1, p2]
    for s, (start, end) in zip(
        body["stops"], [(at(9), at(9, 30)), (at(10), at(10, 30))]
    ):
        reservation = s["reservation"]
        assert reservation["point_id"] == s["point_id"]
        assert reservation["status"] == "active"
        assert reservation["revision"] == 1
        assert reservation["start"] == iso(start)
        assert reservation["end"] == iso(end)
        assert reservation["rules_version"] == 1
        assert reservation["cancel_token"]
        assert reservation["cancelled_at"] is None
    # Each stop got its own reservation number and cancel token.
    ids = [s["reservation"]["id"] for s in body["stops"]]
    tokens = [s["reservation"]["cancel_token"] for s in body["stops"]]
    assert len(set(ids)) == 2
    assert len(set(tokens)) == 2


def test_trip_with_two_and_five_stops_accepted(client):
    points = open_points(client, 5)
    two = book_trip(
        client,
        "two",
        [stop(points[0], at(9), at(9, 30)), stop(points[1], at(9, 30), at(10))],
    )
    assert two.status_code == 201, two.text
    assert len(two.json()["stops"]) == 2
    five = book_trip(
        client,
        "five",
        [
            stop(points[3], at(8), at(8, 30)),
            stop(points[4], at(9), at(9, 30)),
            stop(points[0], at(10), at(10, 30)),
            stop(points[1], at(10, 30), at(11)),
            stop(points[2], at(11), at(11, 30)),
        ],
    )
    assert five.status_code == 201, five.text
    assert len(five.json()["stops"]) == 5


def test_trip_stops_actually_occupy_their_windows(client):
    (p1, p2), _stops, r = two_stop_trip(client)
    assert r.status_code == 201, r.text
    # Both windows are now taken at their points.
    clash1 = reserve(client, p1, "squatter-1", at(9, 15), at(9, 45))
    assert clash1.status_code == 409
    assert clash1.json()["error"]["code"] == "reservation_overlap"
    clash2 = reserve(client, p2, "squatter-2", at(10), at(10, 30))
    assert clash2.status_code == 409
    assert clash2.json()["error"]["code"] == "reservation_overlap"
    # Adjacent (touching) windows remain bookable.
    ok = reserve(client, p1, "neighbor", at(9, 30), at(10))
    assert ok.status_code == 201, ok.text


def test_trip_stop_reservations_carry_each_points_rules_version(client):
    p1 = open_point(client)
    p2 = open_point(client)
    # Publish a second version at point 2; the trip stop must record it.
    weekday = at(0).isoweekday()
    pub = client.post(
        f"/points/{p2}/rules",
        json={
            "expected_version": 1,
            "rules": {
                "weekly": [
                    {"weekday": weekday, "start_time": "07:00", "end_time": "14:00"}
                ],
                "closed_dates": [],
                "makeup": [],
            },
        },
    )
    assert pub.status_code == 201, pub.text
    r = book_trip(
        client,
        "versions",
        [stop(p1, at(9), at(9, 30)), stop(p2, at(10), at(10, 30))],
    )
    assert r.status_code == 201, r.text
    versions = [s["reservation"]["rules_version"] for s in r.json()["stops"]]
    assert versions == [1, 2]


def test_trip_stop_windows_may_cross_local_midnight(client):
    # Overnight weekly rule 20:00-02:00 (UTC): one continuous open interval.
    tomorrow = (datetime.now(UTC) + timedelta(days=1)).date()
    p1 = open_point(
        client,
        weekly=[
            {
                "weekday": tomorrow.isoweekday(),
                "start_time": "20:00",
                "end_time": "02:00",
            }
        ],
    )
    p2 = open_point(client)
    r = book_trip(
        client,
        "overnight",
        [stop(p1, at(23), at(23, 59)), stop(p2, at(9), at(9, 30))],
    )
    # The second stop (09:00 tomorrow) starts before the first one ends
    # (23:59 tomorrow): rejected as out of visit order.
    assert r.status_code == 422
    r = book_trip(
        client,
        "overnight",
        [stop(p2, at(9), at(9, 30)), stop(p1, at(23), at(23, 59))],
    )
    assert r.status_code == 201, r.text
    # A stop fully inside the overnight interval (crossing local midnight).
    p3 = open_point(client)
    p4 = open_point(
        client,
        weekly=[
            {
                "weekday": tomorrow.isoweekday(),
                "start_time": "20:00",
                "end_time": "02:00",
            }
        ],
    )
    r = book_trip(
        client,
        "overnight-2",
        [stop(p3, at(9), at(9, 30)), stop(p4, at(23, 30), at(0, 30, day_offset=2))],
    )
    assert r.status_code == 201, r.text


# ---------------------------------------------------------------------------
# Structural validation of the itinerary
# ---------------------------------------------------------------------------

def test_trip_requires_two_to_five_stops(client):
    p1 = open_point(client)
    one = book_trip(client, "one-stop", [stop(p1, at(9), at(9, 30))])
    assert one.status_code == 422
    assert one.json()["error"]["code"] == "invalid_request"
    points = open_points(client, 6)
    six = book_trip(
        client,
        "six-stops",
        [
            stop(points[0], at(8), at(8, 30)),
            stop(points[1], at(8, 30), at(9)),
            stop(points[2], at(9), at(9, 30)),
            stop(points[3], at(9, 30), at(10)),
            stop(points[4], at(10), at(10, 30)),
            stop(points[5], at(10, 30), at(11)),
        ],
    )
    assert six.status_code == 422
    assert six.json()["error"]["code"] == "invalid_request"


def test_trip_repeated_point_is_invalid_request(client):
    p1, p2 = open_points(client, 2)
    r = book_trip(
        client,
        "dup",
        [stop(p1, at(9), at(9, 30)), stop(p2, at(9, 30), at(10)), stop(p1, at(10), at(10, 30))],
    )
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "invalid_request"
    assert err["details"]["point_id"] == p1
    assert err["details"]["stop_indexes"] == [0, 2]


def test_trip_out_of_order_stops_are_invalid_period(client):
    p1, p2 = open_points(client, 2)
    r = book_trip(
        client,
        "order",
        [stop(p1, at(9), at(10)), stop(p2, at(9, 30), at(10, 30))],
    )
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "invalid_period"
    assert err["details"]["stop_index"] == 1
    assert err["details"]["previous_stop_index"] == 0


def test_trip_touching_stops_are_allowed(client):
    p1, p2 = open_points(client, 2)
    r = book_trip(
        client,
        "touching",
        [stop(p1, at(9), at(9, 30)), stop(p2, at(9, 30), at(10))],
    )
    assert r.status_code == 201, r.text


def test_trip_naive_timestamps_are_invalid_request(client):
    p1, p2 = open_points(client, 2)
    r = client.post(
        "/trips",
        json={
            "idempotency_key": "naive",
            "stops": [
                {"point_id": p1, "start": "2026-10-01T09:00:00", "end": "2026-10-01T09:30:00"},
                {"point_id": p2, "start": iso(at(10)), "end": iso(at(10, 30))},
            ],
        },
    )
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_request"


# ---------------------------------------------------------------------------
# Per-stop failures: nothing is stored, the failing stop is reported
# ---------------------------------------------------------------------------

def assert_nothing_stored(client, pid, start, end):
    """The failed trip left no reservation behind at ``pid``."""
    proof = reserve(client, pid, "proof-free", start, end)
    assert proof.status_code == 201, proof.text


def test_trip_unknown_point_reports_failing_stop(client):
    p1 = open_point(client)
    r = book_trip(
        client,
        "ghost",
        [stop(p1, at(9), at(9, 30)), stop(999999, at(10), at(10, 30))],
    )
    assert r.status_code == 404
    err = r.json()["error"]
    assert err["code"] == "point_not_found"
    assert err["details"]["stop_index"] == 1
    assert err["details"]["point_id"] == 999999
    assert_nothing_stored(client, p1, at(9), at(9, 30))


def test_trip_unpublished_point_reports_failing_stop(client):
    p1 = open_point(client)
    p2 = create_point(client)  # no rules published
    r = book_trip(
        client,
        "unpublished",
        [stop(p1, at(9), at(9, 30)), stop(p2, at(10), at(10, 30))],
    )
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "no_published_version"
    assert err["details"]["stop_index"] == 1
    assert err["details"]["point_id"] == p2
    assert_nothing_stored(client, p1, at(9), at(9, 30))


def test_trip_invalid_window_reports_failing_stop(client):
    p1, p2 = open_points(client, 2)
    # Second stop exceeds the two-hour maximum duration.
    r = book_trip(
        client,
        "too-long",
        [stop(p1, at(9), at(9, 30)), stop(p2, at(10), at(12, 1))],
    )
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "invalid_period"
    assert err["details"]["stop_index"] == 1
    assert err["details"]["point_id"] == p2
    assert_nothing_stored(client, p1, at(9), at(9, 30))


def test_trip_past_start_reports_failing_stop(client):
    p1, p2 = open_points(client, 2)
    past = datetime.now(UTC) - timedelta(minutes=30)
    r = book_trip(
        client,
        "past",
        [
            stop(p1, past, past + timedelta(minutes=30)),
            stop(p2, at(10), at(10, 30)),
        ],
    )
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "invalid_period"
    assert err["details"]["stop_index"] == 0
    assert err["details"]["point_id"] == p1


def test_trip_outside_open_hours_reports_failing_stop(client):
    p1, p2 = open_points(client, 2)  # both open 08:00-12:00 tomorrow
    r = book_trip(
        client,
        "closed",
        [stop(p1, at(9), at(9, 30)), stop(p2, at(13), at(13, 30))],
    )
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "outside_open_hours"
    assert err["details"]["stop_index"] == 1
    assert err["details"]["point_id"] == p2
    assert_nothing_stored(client, p1, at(9), at(9, 30))


def test_trip_occupied_window_reports_failing_stop_and_conflicts(client):
    p1, p2 = open_points(client, 2)
    holder = reserve(client, p2, "holder", at(10), at(11))
    assert holder.status_code == 201, holder.text
    r = book_trip(
        client,
        "occupied",
        [stop(p1, at(9), at(9, 30)), stop(p2, at(10, 30), at(11))],
    )
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "reservation_overlap"
    assert err["details"]["stop_index"] == 1
    assert err["details"]["point_id"] == p2
    assert err["details"]["conflicting_reservation_ids"] == [holder.json()["id"]]
    # The whole group rolled back: the first stop's window is still free.
    assert_nothing_stored(client, p1, at(9), at(9, 30))


def test_trip_failure_does_not_consume_the_idempotency_key(client):
    p1, p2 = open_points(client, 2)
    bad = book_trip(
        client,
        "retry-key",
        [stop(p1, at(9), at(9, 30)), stop(p2, at(13), at(13, 30))],
    )
    assert bad.status_code == 409
    fixed = book_trip(
        client,
        "retry-key",
        [stop(p1, at(9), at(9, 30)), stop(p2, at(10), at(10, 30))],
    )
    assert fixed.status_code == 201, fixed.text


# ---------------------------------------------------------------------------
# Trip idempotency
# ---------------------------------------------------------------------------

def test_same_trip_key_same_content_replays_first_snapshot(client):
    (p1, p2), stops, first = two_stop_trip(client, key="idem-replay")
    assert first.status_code == 201, first.text
    replay = book_trip(client, "idem-replay", stops)
    assert replay.status_code == 200, replay.text
    assert replay.json() == first.json()
    # No stop was booked twice: the windows are held exactly once.
    clash = reserve(client, p1, "double-check", at(9), at(9, 30))
    assert clash.status_code == 409
    assert clash.json()["error"]["code"] == "reservation_overlap"


def test_trip_replay_after_stop_cancel_reports_original_window_live_status(client):
    (p1, p2), stops, first = two_stop_trip(client, key="idem-cancel")
    assert first.status_code == 201, first.text
    first_stop = first.json()["stops"][0]["reservation"]
    cancel(client, p1, first_stop["id"], first_stop["cancel_token"])

    replay = book_trip(client, "idem-cancel", stops)
    assert replay.status_code == 200, replay.text
    body = replay.json()
    # First-creation snapshot: original window / rules version / revision 1 ...
    cancelled_stop = body["stops"][0]["reservation"]
    assert cancelled_stop["id"] == first_stop["id"]
    assert cancelled_stop["revision"] == 1
    assert cancelled_stop["start"] == iso(at(9))
    assert cancelled_stop["end"] == iso(at(9, 30))
    assert cancelled_stop["cancel_token"] == first_stop["cancel_token"]
    # ... but the status reflects the live row.
    assert cancelled_stop["status"] == "cancelled"
    assert cancelled_stop["cancelled_at"] is not None
    assert body["stops"][1]["reservation"]["status"] == "active"


def test_trip_replay_after_stop_reschedule_reports_first_created_window(client):
    (p1, p2), stops, first = two_stop_trip(client, key="idem-reschedule")
    assert first.status_code == 201, first.text
    first_stop = first.json()["stops"][0]["reservation"]
    moved = client.post(
        f"/points/{p1}/reservations/{first_stop['id']}/reschedule",
        json={
            "cancel_token": first_stop["cancel_token"],
            "idempotency_key": "move-stop",
            "expected_revision": 1,
            "start": iso(at(11)),
            "end": iso(at(11, 30)),
        },
    )
    assert moved.status_code == 200, moved.text
    assert moved.json()["revision"] == 2

    replay = book_trip(client, "idem-reschedule", stops)
    assert replay.status_code == 200, replay.text
    replayed_stop = replay.json()["stops"][0]["reservation"]
    assert replayed_stop["revision"] == 1
    assert replayed_stop["start"] == iso(at(9))
    assert replayed_stop["end"] == iso(at(9, 30))
    assert replayed_stop["status"] == "active"


def test_same_trip_key_different_content_conflicts(client):
    (p1, p2), stops, first = two_stop_trip(client, key="idem-conflict")
    assert first.status_code == 201, first.text
    p3 = open_point(client)
    variants = [
        # different window at one stop
        [stop(p1, at(9), at(9, 30)), stop(p2, at(10, 30), at(11))],
        # different point at one stop
        [stop(p1, at(9), at(9, 30)), stop(p3, at(10), at(10, 30))],
        # same stops in a different order (still chronologically valid)
        [stop(p2, at(9), at(9, 30)), stop(p1, at(10), at(10, 30))],
        # different stop count
        [stop(p1, at(9), at(9, 30)), stop(p2, at(10), at(10, 30)), stop(p3, at(11), at(11, 30))],
    ]
    for changed in variants:
        conflict = book_trip(client, "idem-conflict", changed)
        assert conflict.status_code == 409, changed
        err = conflict.json()["error"]
        assert err["code"] == "idempotency_conflict"
        assert err["details"]["trip_id"] == first.json()["trip_id"]
        assert [s["point_id"] for s in err["details"]["stops"]] == [p1, p2]


def test_trip_key_namespace_is_independent_of_reservation_keys(client):
    p1, p2 = open_points(client, 2)
    direct = reserve(client, p1, "shared-key", at(8), at(8, 30))
    assert direct.status_code == 201, direct.text
    trip = book_trip(
        client,
        "shared-key",
        [stop(p1, at(9), at(9, 30)), stop(p2, at(10), at(10, 30))],
    )
    assert trip.status_code == 201, trip.text


# ---------------------------------------------------------------------------
# Trip stops behave as ordinary reservations afterwards
# ---------------------------------------------------------------------------

def test_trip_stop_is_cancellable_and_releases_the_window(client):
    (p1, p2), _stops, r = two_stop_trip(client)
    assert r.status_code == 201, r.text
    first_stop = r.json()["stops"][0]["reservation"]
    cancelled = cancel(client, p1, first_stop["id"], first_stop["cancel_token"])
    assert cancelled.status_code == 200, cancelled.text
    assert cancelled.json()["status"] == "cancelled"
    # The released window is bookable again; the other stop is untouched.
    again = reserve(client, p1, "new-tenant", at(9), at(9, 30))
    assert again.status_code == 201, again.text
    clash = reserve(client, p2, "still-held", at(10), at(10, 30))
    assert clash.status_code == 409


def test_trip_stop_cancel_promotes_the_waitlist(client):
    (p1, p2), _stops, r = two_stop_trip(client)
    assert r.status_code == 201, r.text
    first_stop = r.json()["stops"][0]["reservation"]
    queued = waitlist(client, p1, "w-1", at(9), at(9, 30))
    assert queued.status_code == 201, queued.text
    wid, wtoken = queued.json()["id"], queued.json()["query_token"]

    cancelled = cancel(client, p1, first_stop["id"], first_stop["cancel_token"])
    assert cancelled.status_code == 200, cancelled.text
    state = client.get(
        f"/points/{p1}/waitlist/{wid}", params={"query_token": wtoken}
    ).json()
    assert state["status"] == "promoted"
    assert state["reservation"]["start"] == iso(at(9))


def test_trip_blocks_rule_publish_that_uncovers_a_stop(client):
    (p1, p2), _stops, r = two_stop_trip(client)
    assert r.status_code == 201, r.text
    reservation_id = r.json()["stops"][0]["reservation"]["id"]
    blocked = client.post(
        f"/points/{p1}/rules",
        json={
            "expected_version": 1,
            "rules": {"weekly": [], "closed_dates": [], "makeup": []},
        },
    )
    assert blocked.status_code == 409
    err = blocked.json()["error"]
    assert err["code"] == "reservation_change_conflict"
    assert err["details"]["conflicting_reservation_ids"] == [reservation_id]


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

def test_concurrent_trips_same_window_exactly_one_wins(client):
    p1, p2 = open_points(client, 2)
    barrier = threading.Barrier(2)
    outcomes = []

    def attempt(key):
        barrier.wait()
        r = book_trip(
            client,
            key,
            [stop(p1, at(9), at(9, 30)), stop(p2, at(10), at(10, 30))],
        )
        outcomes.append((r.status_code, r.json()))

    threads = [
        threading.Thread(target=attempt, args=("race-1",)),
        threading.Thread(target=attempt, args=("race-2",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    statuses = sorted(s for s, _ in outcomes)
    assert statuses == [201, 409]
    loser = next(b for s, b in outcomes if s == 409)
    assert loser["error"]["code"] == "reservation_overlap"
    assert loser["error"]["details"]["stop_index"] == 0
    # Exactly one group holds the windows.
    clash = reserve(client, p1, "double-check", at(9), at(9, 30))
    assert clash.status_code == 409


def test_concurrent_same_key_same_content_trip_replays(client):
    p1, p2 = open_points(client, 2)
    stops = [stop(p1, at(9), at(9, 30)), stop(p2, at(10), at(10, 30))]
    barrier = threading.Barrier(2)
    outcomes = []

    def attempt():
        barrier.wait()
        r = book_trip(client, "same-key", stops)
        outcomes.append((r.status_code, r.json()))

    threads = [threading.Thread(target=attempt) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    statuses = sorted(s for s, _ in outcomes)
    assert statuses == [200, 201]
    bodies = [b for _, b in outcomes]
    assert bodies[0] == bodies[1]
    # The group was booked exactly once.
    clash = reserve(client, p2, "double-check", at(10), at(10, 30))
    assert clash.status_code == 409


def test_concurrent_trips_reversed_point_order_no_deadlock(client):
    # Trip A visits (p1, p2), trip B visits (p2, p1) with disjoint windows;
    # ascending lock order lets both complete without deadlock.
    p1, p2 = open_points(client, 2)
    barrier = threading.Barrier(2)
    outcomes = {}

    def trip_a():
        barrier.wait()
        outcomes["a"] = book_trip(
            client,
            "trip-a",
            [stop(p1, at(9), at(9, 30)), stop(p2, at(10), at(10, 30))],
        ).status_code

    def trip_b():
        barrier.wait()
        outcomes["b"] = book_trip(
            client,
            "trip-b",
            [stop(p2, at(8), at(8, 30)), stop(p1, at(11), at(11, 30))],
        ).status_code

    threads = [threading.Thread(target=trip_a), threading.Thread(target=trip_b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive(), "deadlock suspected"
    assert outcomes["a"] == 201
    assert outcomes["b"] == 201


def test_concurrent_trip_and_single_reservation_no_double_occupancy(client):
    p1, p2 = open_points(client, 2)
    barrier = threading.Barrier(2)
    outcomes = {}

    def do_trip():
        barrier.wait()
        r = book_trip(
            client,
            "trip-race-book",
            [stop(p1, at(9), at(9, 30)), stop(p2, at(10), at(10, 30))],
        )
        outcomes["trip"] = (r.status_code, r.json())

    def do_book():
        barrier.wait()
        r = reserve(client, p1, "race", at(9), at(9, 30))
        outcomes["book"] = (r.status_code, r.json())

    threads = [threading.Thread(target=do_trip), threading.Thread(target=do_book)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    trip_status, trip_body = outcomes["trip"]
    book_status, book_body = outcomes["book"]
    # Exactly one of them got the 09:00-09:30 window at point 1.
    if trip_status == 201:
        assert book_status == 409
        assert book_body["error"]["code"] == "reservation_overlap"
    else:
        assert book_status == 201
        assert trip_status == 409
        assert trip_body["error"]["code"] == "reservation_overlap"
        assert trip_body["error"]["details"]["stop_index"] == 0
        # The failed trip booked nothing at the second point either.
        free = reserve(client, p2, "proof-free", at(10), at(10, 30))
        assert free.status_code == 201, free.text


def test_concurrent_trip_and_breaking_publish_exactly_one_wins(client):
    p1, p2 = open_points(client, 2)
    barrier = threading.Barrier(2)
    outcomes = {}

    def do_trip():
        barrier.wait()
        r = book_trip(
            client,
            "trip-race-publish",
            [stop(p1, at(9), at(9, 30)), stop(p2, at(10), at(10, 30))],
        )
        outcomes["trip"] = (r.status_code, r.json())

    def do_publish():
        barrier.wait()
        r = client.post(
            f"/points/{p1}/rules",
            json={
                "expected_version": 1,
                "rules": {"weekly": [], "closed_dates": [], "makeup": []},
            },
        )
        outcomes["publish"] = (r.status_code, r.json())

    threads = [threading.Thread(target=do_trip), threading.Thread(target=do_publish)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    trip_status, trip_body = outcomes["trip"]
    pub_status, pub_body = outcomes["publish"]
    if trip_status == 201:
        # Trip committed first; emptying the calendar would un-cover its stop.
        assert pub_status == 409
        assert pub_body["error"]["code"] == "reservation_change_conflict"
    else:
        # Publish committed first; the stop window is no longer open and the
        # whole group rolled back.
        assert trip_status == 409
        assert trip_body["error"]["code"] == "outside_open_hours"
        assert trip_body["error"]["details"]["stop_index"] == 0
        assert pub_status == 201
        free = reserve(client, p2, "proof-free", at(10), at(10, 30))
        assert free.status_code == 201, free.text


def test_concurrent_trip_and_cancel_at_shared_point_serialize(client):
    # A trip booking at p2 races a cancellation at p2 (which runs the
    # waitlist promotion sweep): both complete without 500s and the final
    # occupancy is consistent.
    p1, p2 = open_points(client, 2)
    holder = reserve(client, p2, "holder", at(8), at(8, 30))
    assert holder.status_code == 201, holder.text
    queued = waitlist(client, p2, "w-1", at(8), at(8, 30))
    assert queued.status_code == 201, queued.text
    barrier = threading.Barrier(2)
    outcomes = {}

    def do_trip():
        barrier.wait()
        r = book_trip(
            client,
            "trip-race-cancel",
            [stop(p1, at(9), at(9, 30)), stop(p2, at(10), at(10, 30))],
        )
        outcomes["trip"] = r.status_code

    def do_cancel():
        barrier.wait()
        r = cancel(client, p2, holder.json()["id"], holder.json()["cancel_token"])
        outcomes["cancel"] = r.status_code

    threads = [threading.Thread(target=do_trip), threading.Thread(target=do_cancel)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive(), "deadlock suspected"
    assert outcomes["trip"] == 201
    assert outcomes["cancel"] == 200
    # The release promoted the queued entry exactly once.
    state = client.get(
        f"/points/{p2}/waitlist/{queued.json()['id']}",
        params={"query_token": queued.json()["query_token"]},
    ).json()
    assert state["status"] == "promoted"
