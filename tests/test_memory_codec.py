"""Codec 编解码检查，不依赖仓库或外部服务。"""

from copy import deepcopy
from datetime import timedelta, timezone
import json
import unittest

from context_lab.memory import MemoryCodec, MemoryIndexEvent, MemorySnapshot, MemorySourceRef
from tests.memory_support import NOW, document, record, wire_record


def wire_event(**changes):
    return {
        "id": "event-1", "user_id": "alice", "memory_id": "m1", "operation": "upsert",
        "version": 1, "created_at": NOW.isoformat(), "attempts": 0, "error_code": None,
        **changes,
    }


class MemoryDecodeTests(unittest.TestCase):
    def test_restores_record_references_and_event_without_changing_input(self):
        raw = wire_record()
        raw.update(
            layer="summary", source_refs=[{"memory_id": "raw-1", "version": 2}],
            created_at="2026-11-01T20:00:00+08:00", updated_at="2026-11-01T20:00:00+08:00",
            valid_from=NOW.isoformat(), valid_to=(NOW + timedelta(days=1)).isoformat(),
            metadata={"tags": ["database"]},
        )
        payload = document(raw, revision=3)
        payload["index_events"] = [wire_event(created_at="2026-11-01T20:00:00+08:00")]
        before = deepcopy(payload)
        snapshot = MemoryCodec().decode(payload)
        expected = record(
            layer="summary", source_refs=[MemorySourceRef("raw-1", 2)],
            valid_from=NOW, valid_to=NOW + timedelta(days=1), metadata={"tags": ["database"]},
        )
        self.assertEqual(snapshot, MemorySnapshot(3, [expected], [
            MemoryIndexEvent("event-1", "alice", "m1", "upsert", 1, NOW),
        ]))
        self.assertEqual(snapshot.records[0].created_at.utcoffset(), timedelta(0))
        self.assertEqual(payload, before)
        snapshot.records[0].metadata["tags"].append("new")
        snapshot.records[0].source_refs[0].version = 3
        self.assertEqual(payload, before)

    def test_empty_document_restores_empty_snapshot(self):
        self.assertEqual(MemoryCodec().decode(MemoryCodec.empty_document()), MemorySnapshot())

    def test_rejects_bad_document_shape_and_versions(self):
        cases = [None, [], {}, {**document(), "schema_version": 2},
                 {**document(), "schema_version": True}, {**document(), "unexpected": 1}]
        for key in document():
            missing = document()
            del missing[key]
            cases.append(missing)
        cases += [{**document(), "revision": value} for value in (True, -1, 0.5, "1", None)]
        cases += [{**document(), key: value} for key in ("records", "index_events")
                  for value in ({}, None, "[]")]
        for payload in cases:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                MemoryCodec().decode(payload)

    def test_missing_record_fields_cannot_silently_generate_default_times(self):
        for key in wire_record():
            raw = wire_record()
            del raw[key]
            with self.subTest(field=key), self.assertRaises(ValueError):
                MemoryCodec().decode(document(raw))
        for raw in (None, [], {**wire_record(), "timestamp": NOW.isoformat()}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                MemoryCodec().decode(document(raw))

    def test_rejects_invalid_record_fields(self):
        cases = [(key, value) for key in ("id", "content", "user_id", "memory_type")
                 for value in (None, " ", 3)]
        cases += [(key, value) for key in ("task_id", "fact_key", "supersedes_id")
                  for value in (" ", 3)]
        cases += [("importance", value) for value in (True, -0.1, 1.1, float("nan"), float("inf"), "0.5")]
        cases += [("version", value) for value in (True, 0, -1, "1")]
        cases += [("layer", "other"), ("layer", "working"), ("summary_state", "other"),
                  ("metadata", []), ("metadata", {"bad": float("nan")}),
                  ("metadata", {1: "non-string key"}), ("metadata", {"bad": (1, 2)})]
        for key, value in cases:
            raw = {**wire_record(), key: value}
            with self.subTest(field=key, value=value), self.assertRaises(ValueError):
                MemoryCodec().decode(document(raw))
        working = {**wire_record(), "layer": "working", "task_id": "task-1", "importance": 0}
        self.assertEqual(MemoryCodec().decode(document(working)).records[0].importance, 0)

    def test_rejects_missing_timezone_bad_times_and_reversed_intervals(self):
        for key in ("created_at", "updated_at", "valid_from", "valid_to"):
            for value in ("not-a-date", "2026-11-01T12:00:00", "2026-11-01", 123):
                raw = {**wire_record(), key: value}
                with self.subTest(field=key, value=value), self.assertRaisesRegex(ValueError, key):
                    MemoryCodec().decode(document(raw))
        for key in ("created_at", "updated_at"):
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, key):
                MemoryCodec().decode(document({**wire_record(), key: None}))
        raw = {**wire_record(), "updated_at": (NOW - timedelta(seconds=1)).isoformat()}
        with self.assertRaises(ValueError):
            MemoryCodec().decode(document(raw))
        for end in (NOW, NOW - timedelta(seconds=1)):
            raw = {**wire_record(), "valid_from": NOW.isoformat(), "valid_to": end.isoformat()}
            with self.subTest(end=end), self.assertRaises(ValueError):
                MemoryCodec().decode(document(raw))

    def test_rejects_bad_source_reference_shapes_and_versions(self):
        cases = [None, {}, [None], [{}], [{"memory_id": "raw"}],
                 [{"memory_id": "raw", "version": 1, "extra": True}]]
        cases += [[{"memory_id": "raw", "version": value}] for value in (True, 0, -1, "1")]
        cases += [[{"memory_id": value, "version": 1}] for value in (None, " ", 3)]
        for refs in cases:
            raw = {**wire_record(), "source_refs": refs}
            with self.subTest(refs=refs), self.assertRaises(ValueError):
                MemoryCodec().decode(document(raw))

    def test_rejects_bad_event_fields_with_clear_errors(self):
        cases = [None, [], {**wire_event(), "extra": 1}]
        for key in wire_event():
            missing = wire_event()
            del missing[key]
            cases.append(missing)
        cases += [wire_event(**{key: value}) for key in ("id", "user_id", "memory_id")
                  for value in (None, " ", 3)]
        cases += [wire_event(operation="other"), wire_event(version=True), wire_event(version=0),
                  wire_event(attempts=True), wire_event(attempts=-1), wire_event(attempts="1"),
                  wire_event(created_at=NOW.replace(tzinfo=None).isoformat()),
                  wire_event(created_at=None), wire_event(error_code=False), wire_event(error_code=" ")]
        for event in cases:
            payload = document()
            payload["index_events"] = [event]
            with self.subTest(event=event), self.assertRaisesRegex(ValueError, "index_events"):
                MemoryCodec().decode(payload)

    def test_rejects_duplicate_record_and_event_ids_within_a_user(self):
        with self.assertRaises(ValueError):
            MemoryCodec().decode(document(wire_record(), wire_record()))
        payload = document()
        payload["index_events"] = [wire_event(), wire_event()]
        with self.assertRaises(ValueError):
            MemoryCodec().decode(payload)

    def test_keeps_all_users_and_allows_cleanup_events_and_stale_evidence(self):
        raw = {**wire_record(), "layer": "summary", "summary_state": "stale",
               "source_refs": [{"memory_id": "deleted-source", "version": 1}]}
        payload = document(raw, wire_record(user_id="bob"))
        payload["index_events"] = [wire_event(operation="delete", memory_id="deleted"),
                                   wire_event(user_id="bob")]
        snapshot = MemoryCodec().decode(payload)
        self.assertEqual([item.user_id for item in snapshot.records], ["alice", "bob"])
        self.assertEqual(snapshot.index_events[0].memory_id, "deleted")
        self.assertEqual(snapshot.records[0].source_refs[0].memory_id, "deleted-source")

    def test_accepts_utc_z_suffix(self):
        raw = {**wire_record(), "created_at": "2026-11-01T12:00:00Z"}
        self.assertEqual(MemoryCodec().decode(document(raw)).records[0].created_at, NOW)


class MemoryEncodeTests(unittest.TestCase):
    def test_json_round_trip_preserves_records_events_and_original_snapshot(self):
        items = [record(metadata={"tags": ["database"]}), record(
            "summary-1", layer="summary", source_refs=[MemorySourceRef("m1", 1)],
            valid_from=NOW, valid_to=NOW + timedelta(days=1),
        )]
        event = MemoryIndexEvent("event-1", "alice", "m1", "upsert", 1, NOW)
        snapshot = MemorySnapshot(3, items, [event])
        before = deepcopy(snapshot)
        payload = MemoryCodec().encode(snapshot)
        self.assertEqual(snapshot, before)
        restored = json.loads(json.dumps(payload, allow_nan=False))
        self.assertEqual(MemoryCodec().decode(restored), snapshot)
        self.assertEqual(set(payload), {"schema_version", "revision", "records", "index_events"})
        self.assertIsNone(payload["records"][0]["valid_from"])
        self.assertIsNone(payload["records"][0]["valid_to"])
        self.assertEqual(payload["records"][1]["source_refs"], [{"memory_id": "m1", "version": 1}])
        payload["records"][0]["metadata"]["tags"].append("new")
        payload["records"][1]["source_refs"][0]["version"] = 2
        payload["index_events"][0]["attempts"] = 9
        self.assertEqual(snapshot, before)

    def test_empty_snapshot_produces_default_document(self):
        self.assertEqual(MemoryCodec().encode(MemorySnapshot()), MemoryCodec.empty_document())

    def test_normalizes_record_and_event_times_to_utc_without_changing_input(self):
        local = NOW.astimezone(timezone(timedelta(hours=8)))
        item = record(created_at=local, updated_at=local, valid_from=local,
                      valid_to=local + timedelta(days=1))
        event = MemoryIndexEvent("event-1", "alice", "m1", "upsert", 1, local)
        snapshot = MemorySnapshot(1, [item], [event])
        payload = MemoryCodec().encode(snapshot)
        for key in ("created_at", "updated_at", "valid_from"):
            self.assertEqual(payload["records"][0][key], NOW.isoformat())
        self.assertEqual(payload["records"][0]["valid_to"], (NOW + timedelta(days=1)).isoformat())
        self.assertEqual(payload["index_events"][0]["created_at"], NOW.isoformat())
        self.assertEqual(snapshot.records[0].created_at.utcoffset(), timedelta(hours=8))

    def test_rejects_invalid_typed_containers(self):
        cases = [None, {}, MemorySnapshot(records={}), MemorySnapshot(index_events={}),
                 MemorySnapshot(records=[wire_record()]), MemorySnapshot(index_events=[wire_event()]),
                 MemorySnapshot(records=[record(source_refs={})]),
                 MemorySnapshot(records=[record(source_refs=[{"memory_id": "m1", "version": 1}])])]
        for snapshot in cases:
            with self.subTest(snapshot=snapshot), self.assertRaises(ValueError):
                MemoryCodec().encode(snapshot)

    def test_rejects_naive_times_strings_and_missing_required_times(self):
        for key in ("created_at", "updated_at", "valid_from", "valid_to"):
            for value in (NOW.replace(tzinfo=None), NOW.isoformat(), 123):
                snapshot = MemorySnapshot(records=[record(**{key: value})])
                before = deepcopy(snapshot)
                with self.subTest(field=key, value=value), self.assertRaisesRegex(ValueError, key):
                    MemoryCodec().encode(snapshot)
                self.assertEqual(snapshot, before)
        for key in ("created_at", "updated_at"):
            with self.subTest(field=key), self.assertRaisesRegex(ValueError, key):
                MemoryCodec().encode(MemorySnapshot(records=[record(**{key: None})]))
        for value in (None, NOW.replace(tzinfo=None), NOW.isoformat()):
            event = MemoryIndexEvent("event-1", "alice", "m1", "upsert", 1, value)
            with self.subTest(event_time=value), self.assertRaisesRegex(ValueError, "index_events"):
                MemoryCodec().encode(MemorySnapshot(index_events=[event]))

    def test_rejects_unreadable_payloads_before_returning(self):
        cases = [MemorySnapshot(revision=True), MemorySnapshot(revision=-1),
                 MemorySnapshot(records=[record(importance=2)]),
                 MemorySnapshot(records=[record(), record()]),
                 MemorySnapshot(records=[record(metadata={"bad": float("inf")})]),
                 MemorySnapshot(records=[record(updated_at=NOW - timedelta(seconds=1))]),
                 MemorySnapshot(records=[record(valid_from=NOW, valid_to=NOW)])]
        for snapshot in cases:
            with self.subTest(snapshot=snapshot), self.assertRaises(ValueError):
                MemoryCodec().encode(snapshot)
