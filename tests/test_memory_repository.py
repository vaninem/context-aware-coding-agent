"""仓库读写检查：用户隔离、原子保存和并发版本冲突。"""

from copy import deepcopy
import json
import multiprocessing
from unittest.mock import patch

from context_lab.memory import JsonMemoryRepository, MemoryCodec, MemoryConflict, MemoryIndexEvent, MemorySnapshot
from tests.memory_support import NOW, record
from tests.support import SandboxTestCase


def _commit_in_process(path, memory_id, barrier, outcomes):
    """两个独立进程先读同一版本，再竞争提交；只操作临时业务文件。"""
    try:
        repository = JsonMemoryRepository(path, "alice")
        snapshot = repository.read_snapshot()
        snapshot.records.append(record(memory_id))
        barrier.wait(timeout=10)
        result = repository.commit(snapshot)
        outcomes.put(("saved", memory_id, result.revision))
    except MemoryConflict:
        outcomes.put(("conflict", memory_id, None))
    except Exception as error:
        outcomes.put(("error", memory_id, f"{type(error).__name__}: {error}"))


class MemoryRepositoryReadTests(SandboxTestCase):
    def write_snapshot(self, path, revision, records, events):
        payload = MemoryCodec().encode(MemorySnapshot(revision, records, events))
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_filters_events_and_records_but_keeps_file_revision(self):
        path = self.tmp / "memory.json"
        alice, bob = record("a"), record("b", user_id="bob")
        alice_event = MemoryIndexEvent("ea", "alice", "a", "upsert", 1, NOW)
        bob_event = MemoryIndexEvent("eb", "bob", "b", "upsert", 1, NOW)
        self.write_snapshot(path, 7, [alice, bob], [alice_event, bob_event])
        repository = JsonMemoryRepository(path, "alice")
        before = path.read_bytes()
        with patch.object(repository.store, "write", side_effect=AssertionError("读取不能写文件")):
            snapshot = repository.read_snapshot()
        self.assertEqual(snapshot, MemorySnapshot(7, [alice], [alice_event]))
        self.assertEqual(JsonMemoryRepository(path, "charlie").read_snapshot(), MemorySnapshot(7))
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(repository.lock_path.exists())

    def test_repeated_reads_see_external_updates_and_return_independent_objects(self):
        path = self.tmp / "memory.json"
        self.write_snapshot(path, 1, [record(metadata={"tags": ["original"]})], [])
        repository = JsonMemoryRepository(path, "alice")
        first = repository.read_snapshot()
        first.records[0].metadata["tags"].append("local change")
        second = JsonMemoryRepository(path, "alice").read_snapshot()
        self.assertEqual(second.records[0].metadata["tags"], ["original"])
        self.write_snapshot(path, 2, [record(content="updated elsewhere")], [])
        third = repository.read_snapshot()
        self.assertEqual(third.revision, 2)
        self.assertEqual(third.records[0].content, "updated elsewhere")
        self.assertEqual(first.revision, 1)
        self.assertEqual(first.records[0].content, "redis timeout")

    def test_corrupt_json_or_document_is_not_treated_as_empty_memory(self):
        path = self.tmp / "memory.json"
        repository = JsonMemoryRepository(path, "alice")
        for text in ('{"records":', 'null', '[]', '{}'):
            with self.subTest(text=text):
                path.write_text(text, encoding="utf-8")
                before = path.read_bytes()
                with self.assertRaises(ValueError):
                    repository.read_snapshot()
                self.assertEqual(path.read_bytes(), before)

    def test_file_read_errors_are_not_swallowed(self):
        repository = JsonMemoryRepository(self.tmp / "memory.json", "alice")
        with patch.object(repository.store, "read", side_effect=PermissionError("test read denied")):
            with self.assertRaises(PermissionError):
                repository.read_snapshot()


class MemoryRepositoryCommitTests(SandboxTestCase):
    def repository(self, user_id="alice"):
        return JsonMemoryRepository(self.tmp / "memory.json", user_id)

    def test_first_commit_saves_record_and_event_and_returns_independent_copy(self):
        repository = self.repository()
        item = record(metadata={"tags": ["original"]})
        event = MemoryIndexEvent("event-1", "alice", item.id, "upsert", 1, NOW)
        snapshot = MemorySnapshot(0, [item], [event])
        before = deepcopy(snapshot)
        returned = repository.commit(snapshot)
        self.assertEqual(snapshot, before)
        self.assertEqual(returned, MemorySnapshot(1, [item], [event]))
        self.assertEqual(self.repository().read_snapshot(), returned)
        returned.records[0].metadata["tags"].append("returned change")
        returned.index_events[0].attempts = 3
        snapshot.records[0].content = "input change"
        snapshot.index_events[0].error_code = "input_error"
        self.assertEqual(self.repository().read_snapshot(), MemorySnapshot(1, before.records, before.index_events))
        self.assertEqual(snapshot.records[0].metadata["tags"], ["original"])
        self.assertIsNone(returned.index_events[0].error_code)

    def test_replaces_user_partition_preserves_others_and_returns_only_bound_user(self):
        alice, bob = self.repository(), self.repository("bob")
        a, b = record("a"), record("b", user_id="bob")
        ea = MemoryIndexEvent("ea", "alice", "a", "upsert", 1, NOW)
        eb = MemoryIndexEvent("eb", "bob", "b", "upsert", 1, NOW)
        alice.commit(MemorySnapshot(0, [a], [ea]))
        bob.commit(MemorySnapshot(1, [b], [eb]))
        changed = record("a", content="updated")
        ea_new = MemoryIndexEvent("ea-new", "alice", "a", "upsert", 1, NOW)
        returned = alice.commit(MemorySnapshot(2, [changed], [ea_new]))
        self.assertEqual(returned, MemorySnapshot(3, [changed], [ea_new]))
        self.assertEqual(bob.read_snapshot(), MemorySnapshot(3, [b], [eb]))
        # 空快照只清空当前用户分区，旧记录和旧事件都不能残留。
        self.assertEqual(alice.commit(MemorySnapshot(3)), MemorySnapshot(4))
        self.assertEqual(alice.read_snapshot(), MemorySnapshot(4))
        self.assertEqual(bob.read_snapshot(), MemorySnapshot(4, [b], [eb]))

    def test_foreign_records_and_events_are_rejected_without_changing_file(self):
        repository = self.repository()
        repository.commit(MemorySnapshot(records=[record()]))
        before = repository.path.read_bytes()
        bad_event = MemoryIndexEvent("bad-event", "bob", "b", "delete", 1, NOW)
        for snapshot in (MemorySnapshot(1, [record("b", user_id="bob")]),
                         MemorySnapshot(1, [record()], [bad_event])):
            with self.subTest(snapshot=snapshot), self.assertRaises(ValueError):
                repository.commit(snapshot)
            self.assertEqual(repository.path.read_bytes(), before)

    def test_invalid_snapshots_and_mismatched_revisions_cannot_write(self):
        repository = self.repository()
        repository.commit(MemorySnapshot(records=[record()]))
        before = repository.path.read_bytes()
        invalid = [None, {}, MemorySnapshot(True), MemorySnapshot(-1),
                   MemorySnapshot(1, [record(importance=2)])]
        for snapshot in invalid:
            with self.subTest(snapshot=snapshot), self.assertRaises(ValueError):
                repository.commit(snapshot)
            self.assertEqual(repository.path.read_bytes(), before)
        for revision in (0, 2):
            with self.subTest(revision=revision), self.assertRaises(MemoryConflict):
                repository.commit(MemorySnapshot(revision))
            self.assertEqual(repository.path.read_bytes(), before)

    def test_atomic_replace_failure_preserves_file_and_input_and_releases_lock(self):
        repository = self.repository()
        repository.commit(MemorySnapshot(records=[record()]))
        incoming = repository.read_snapshot()
        incoming.records[0].content = "changed"
        incoming.index_events.append(MemoryIndexEvent("event-2", "alice", "m1", "upsert", 1, NOW))
        before_input = deepcopy(incoming)
        before_file = repository.path.read_bytes()
        with patch("context_lab.storage.json_store.os.replace", side_effect=OSError("test replace failed")):
            with self.assertRaises(OSError):
                repository.commit(incoming)
        self.assertEqual(repository.path.read_bytes(), before_file)
        self.assertEqual(incoming, before_input)
        self.assertEqual(list(self.tmp.glob(".memory.json.*.tmp")), [])
        # 失败没有消费版本或遗留持锁状态；同一个输入仍可成功提交。
        returned = repository.commit(incoming)
        self.assertEqual(returned.revision, 2)
        self.assertEqual(repository.read_snapshot(), returned)

    def test_corrupt_existing_file_is_not_overwritten(self):
        repository = self.repository()
        repository.path.write_text('{"unfinished":', encoding="utf-8")
        before = repository.path.read_bytes()
        with self.assertRaises(ValueError):
            repository.commit(MemorySnapshot(records=[record()]))
        self.assertEqual(repository.path.read_bytes(), before)

    def test_acknowledge_retries_with_latest_records_and_preserves_new_events(self):
        repository = self.repository()
        completed = MemoryIndexEvent("e1", "alice", "m1", "upsert", 1, NOW)
        initial = repository.commit(MemorySnapshot(records=[record()], index_events=[completed]))
        updated = record(content="changed during acknowledgement", version=2)
        new_event = MemoryIndexEvent("e2", "alice", "m1", "upsert", 2, NOW)
        commit = repository.commit

        def commit_after_concurrent_update(snapshot):
            if snapshot.revision == initial.revision:
                self.repository().commit(MemorySnapshot(initial.revision, [updated], [completed, new_event]))
            return commit(snapshot)

        with patch.object(repository, "commit", side_effect=commit_after_concurrent_update) as commits:
            repository.acknowledge([completed.id])

        self.assertEqual(commits.call_count, 2)
        self.assertEqual(repository.read_snapshot(), MemorySnapshot(initial.revision + 2, [updated], [new_event]))

    def test_two_processes_using_same_revision_have_one_winner(self):
        context = multiprocessing.get_context("spawn")
        outcomes = context.Queue()
        barrier = context.Barrier(2)
        repository = self.repository()
        processes = [context.Process(target=_commit_in_process,
                                     args=(repository.path, memory_id, barrier, outcomes))
                     for memory_id in ("a", "b")]
        try:
            for process in processes:
                process.start()
            results = [outcomes.get(timeout=15) for _ in processes]
            for process in processes:
                process.join(timeout=5)
                self.assertEqual(process.exitcode, 0)
            self.assertCountEqual([result[0] for result in results], ["saved", "conflict"])
            winner = next(result[1] for result in results if result[0] == "saved")
            snapshot = repository.read_snapshot()
            self.assertEqual(snapshot.revision, 1)
            self.assertEqual([item.id for item in snapshot.records], [winner])
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                if process.pid is not None:
                    process.join(timeout=5)
            outcomes.close()
            outcomes.join_thread()
