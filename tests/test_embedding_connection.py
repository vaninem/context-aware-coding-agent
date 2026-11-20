"""连接接线和资源清理测试；全部使用模拟连接，不连接真实数据库。"""

import unittest
from unittest.mock import Mock, patch

from context_lab.embeddings import PostgresEmbeddingStore
from context_lab.embeddings.errors import EmbeddingStorageError


class EmbeddingConnectionTests(unittest.TestCase):
    def setUp(self):
        self.url = "postgresql://test:fake-test-password@invalid.example/test"
        self.conn = Mock()
        self.factory = Mock(return_value=self.conn)
        self.store = PostgresEmbeddingStore(self.url, connect_timeout=7, connection_factory=self.factory)

    def test_injected_connection_is_registered_and_returned_open(self):
        with patch("context_lab.embeddings.store.register_vector") as register, patch("psycopg.connect") as connect:
            self.assertIs(self.store.connect(), self.conn)
        self.factory.assert_called_once_with(self.url, connect_timeout=7)
        register.assert_called_once_with(self.conn)
        connect.assert_not_called()
        self.conn.close.assert_not_called()

    def test_default_factory_is_psycopg_connect(self):
        store = PostgresEmbeddingStore(self.url)
        with patch("psycopg.connect", return_value=self.conn) as connect, patch("context_lab.embeddings.store.register_vector"):
            self.assertIs(store.connect(), self.conn)
        connect.assert_called_once_with(self.url, connect_timeout=10)

    def test_each_connection_gets_its_own_vector_registration(self):
        other = Mock()
        self.factory.side_effect = [self.conn, other]
        with patch("context_lab.embeddings.store.register_vector") as register:
            self.store.connect()
            self.store.connect()
        self.assertEqual(register.call_args_list[0].args, (self.conn,))
        self.assertEqual(register.call_args_list[1].args, (other,))

    def test_registration_failure_closes_connection_and_redacts_error(self):
        with patch("context_lab.embeddings.store.register_vector", side_effect=RuntimeError(self.url)):
            with self.assertRaises(EmbeddingStorageError) as error:
                self.store.connect()
        self.conn.close.assert_called_once_with()
        self.assertNotIn("fake-test-password", str(error.exception))
        self.assertIn("RuntimeError", str(error.exception))
        self.assertTrue(error.exception.__suppress_context__)

    def test_close_failure_does_not_mask_registration_failure(self):
        self.conn.close.side_effect = OSError("fake-test-password")
        with patch("context_lab.embeddings.store.register_vector", side_effect=ValueError(self.url)):
            with self.assertRaises(EmbeddingStorageError) as error:
                self.store.connect()
        self.assertIn("ValueError", str(error.exception))
        self.assertNotIn("OSError", str(error.exception))
        self.assertNotIn("fake-test-password", str(error.exception))

    def test_connection_failure_is_redacted_before_registration(self):
        self.factory.side_effect = OSError(self.url)
        with patch("context_lab.embeddings.store.register_vector") as register:
            with self.assertRaises(EmbeddingStorageError) as error:
                self.store.connect()
        register.assert_not_called()
        self.assertNotIn("fake-test-password", str(error.exception))

    def test_invalid_configuration_stops_before_connection(self):
        cases = [("", 10), (" ", 10), (None, 10), (self.url, 0), (self.url, True)]
        for url, timeout in cases:
            store = PostgresEmbeddingStore(url, connect_timeout=timeout, connection_factory=self.factory)
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                store.connect()
        self.factory.assert_not_called()
