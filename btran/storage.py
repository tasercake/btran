"""Durable v2 SQLite and deterministic sealed-revision storage.

The v2 store is intentionally small.  SQLite is the mutable index/record
store; sealed revisions are standalone ZIP snapshots and are the authority for
cache selection.  All bytes entering the store are canonical UTF-8 JSON.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

from btran.schema import canonical_json_bytes


class StorageError(ValueError):
    """Invalid, conflicting, or corrupt v2 storage."""


DB_NAME = "state-v2.sqlite3"


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise StorageError(f"{name} must be a non-empty string")
    return value


def _ids(values: Sequence[str], name: str) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple, set, frozenset)):
        raise StorageError(f"{name} must be a sequence")
    result = tuple(_text(v, name) for v in values)
    if len(set(result)) != len(result):
        raise StorageError(f"{name} must be unique")
    return tuple(sorted(result))


_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
 record_id TEXT PRIMARY KEY, kind TEXT NOT NULL, canonical_json_bytes BLOB NOT NULL,
 content_sha256 TEXT NOT NULL, semantic_key TEXT
);
CREATE TABLE IF NOT EXISTS findings (
 finding_id TEXT PRIMARY KEY, canonical_json_bytes BLOB NOT NULL, content_sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS record_dependencies (
 record_id TEXT NOT NULL, dependency_id TEXT NOT NULL,
 PRIMARY KEY(record_id, dependency_id)
);
CREATE TABLE IF NOT EXISTS record_findings (
 record_id TEXT NOT NULL, finding_id TEXT NOT NULL,
 PRIMARY KEY(record_id, finding_id)
);
CREATE TABLE IF NOT EXISTS edges (
 edge_id TEXT PRIMARY KEY, canonical_json_bytes BLOB NOT NULL, content_sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attestations (
 attestation_id TEXT PRIMARY KEY, canonical_json_bytes BLOB NOT NULL, content_sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS semantic_index (
 semantic_key TEXT NOT NULL, record_id TEXT NOT NULL,
 PRIMARY KEY(semantic_key, record_id)
);
CREATE TABLE IF NOT EXISTS revisions (
 revision_id TEXT PRIMARY KEY, zip_filename TEXT NOT NULL, zip_sha256 TEXT NOT NULL,
 canonical_snapshot_json BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS active_revision (
 slot INTEGER PRIMARY KEY CHECK(slot=1), revision_id TEXT NOT NULL
);
"""


class Storage:
    """SQLite v2 state with one durable transaction protocol for all writes."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / DB_NAME
        self.revisions_dir = self.root / "revisions"
        self.revisions_dir.mkdir(parents=True, exist_ok=True)
        existed = self.path.exists()
        self._initialize(existed)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        # These are checked on every writer connection before BEGIN IMMEDIATE.
        journal = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if journal.lower() != "wal":
            connection.close()
            raise StorageError("v2 storage requires SQLite WAL mode")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA wal_autocheckpoint=0")
        values = {
            "journal_mode": connection.execute("PRAGMA journal_mode").fetchone()[0],
            "synchronous": connection.execute("PRAGMA synchronous").fetchone()[0],
            "wal_autocheckpoint": connection.execute("PRAGMA wal_autocheckpoint").fetchone()[0],
        }
        if values != {"journal_mode": "wal", "synchronous": 2, "wal_autocheckpoint": 0}:
            connection.close()
            raise StorageError(f"unsafe SQLite durability settings: {values!r}")
        return connection

    def _initialize(self, existed: bool) -> None:
        connection = self._connect()
        try:
            connection.executescript(_SCHEMA)
            connection.commit()
            connection.execute("PRAGMA wal_checkpoint(FULL)")
        finally:
            connection.close()
        self._fsync_db()
        if not existed:
            _fsync_dir(self.root)

    def _fsync_db(self) -> None:
        with self.path.open("rb") as handle:
            os.fsync(handle.fileno())
        wal = self.path.with_name(self.path.name + "-wal")
        if wal.exists():
            with wal.open("rb") as handle:
                os.fsync(handle.fileno())

    def _write(self, operation: Callable[[sqlite3.Connection], Any]) -> Any:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            result = operation(connection)
            connection.commit()
            checkpoint = connection.execute("PRAGMA wal_checkpoint(FULL)").fetchone()
            # SQLITE_BUSY here means the transaction was not durably flushed.
            if checkpoint is not None and int(checkpoint[0]) != 0:
                raise StorageError("SQLite WAL checkpoint was not successful")
            self._fsync_db()
            return result
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _insert_immutable(connection: sqlite3.Connection, table: str, key: str,
                          key_column: str, data: bytes, sha_column: str = "content_sha256") -> None:
        if not isinstance(data, bytes):
            raise StorageError("canonical record data must be bytes")
        digest = _sha(data)
        row = connection.execute(
            f"SELECT {key_column}, canonical_json_bytes, {sha_column} FROM {table} WHERE {key_column}=?", (key,)
        ).fetchone()
        if row is not None:
            if bytes(row["canonical_json_bytes"]) != data or row[sha_column] != digest:
                raise StorageError(f"immutable {table} ID has conflicting bytes")
            return
        connection.execute(
            f"INSERT INTO {table}({key_column}, canonical_json_bytes, {sha_column}) VALUES(?,?,?)",
            (key, data, digest),
        )

    def put_record(self, record_id: str, kind: str, data: bytes, *, semantic_key: str | None = None,
                   dependency_ids: Sequence[str] = (), finding_ids: Sequence[str] = ()) -> None:
        record_id, kind = _text(record_id, "record_id"), _text(kind, "kind")
        dependencies, findings = _ids(dependency_ids, "dependency_ids"), _ids(finding_ids, "finding_ids")
        if semantic_key is not None:
            _text(semantic_key, "semantic_key")

        def operation(connection: sqlite3.Connection) -> None:
            digest = _sha(data)
            row = connection.execute("SELECT kind, canonical_json_bytes, content_sha256, semantic_key FROM records WHERE record_id=?", (record_id,)).fetchone()
            if row is not None:
                if (row["kind"] != kind or bytes(row["canonical_json_bytes"]) != data
                        or row["content_sha256"] != digest):
                    raise StorageError("immutable record ID has conflicting content")
            else:
                connection.execute("INSERT INTO records VALUES(?,?,?,?,?)", (record_id, kind, data, digest, semantic_key))
            for dependency_id in dependencies:
                connection.execute("INSERT OR IGNORE INTO record_dependencies VALUES(?,?)", (record_id, dependency_id))
            for finding_id in findings:
                connection.execute("INSERT OR IGNORE INTO record_findings VALUES(?,?)", (record_id, finding_id))
            if semantic_key is not None:
                connection.execute("INSERT OR IGNORE INTO semantic_index VALUES(?,?)", (semantic_key, record_id))

        self._write(operation)

    def put_finding(self, finding_id: str, data: bytes) -> None:
        finding_id = _text(finding_id, "finding_id")
        self._write(lambda c: self._insert_immutable(c, "findings", finding_id, "finding_id", data))

    def put_edge(self, edge_id: str, data: bytes) -> None:
        edge_id = _text(edge_id, "edge_id")
        self._write(lambda c: self._insert_immutable(c, "edges", edge_id, "edge_id", data))

    def put_attestation(self, attestation_id: str, data: bytes) -> None:
        attestation_id = _text(attestation_id, "attestation_id")
        self._write(lambda c: self._insert_immutable(c, "attestations", attestation_id, "attestation_id", data))

    def _fetch(self, table: str, key_column: str, key: str) -> bytes:
        connection = self._connect()
        try:
            row = connection.execute(f"SELECT canonical_json_bytes FROM {table} WHERE {key_column}=?", (key,)).fetchone()
            if row is None:
                raise StorageError(f"missing {table} value {key}")
            data = bytes(row[0])
            digest = _sha(data)
            stored = connection.execute(f"SELECT content_sha256 FROM {table} WHERE {key_column}=?", (key,)).fetchone()[0]
            if digest != stored:
                raise StorageError(f"{table} content hash mismatch: {key}")
            return data
        finally:
            connection.close()

    def record_bytes(self, record_id: str) -> bytes:
        return self._fetch("records", "record_id", record_id)

    def finding_bytes(self, finding_id: str) -> bytes:
        return self._fetch("findings", "finding_id", finding_id)

    def edge_bytes(self, edge_id: str) -> bytes:
        return self._fetch("edges", "edge_id", edge_id)

    def attestation_bytes(self, attestation_id: str) -> bytes:
        return self._fetch("attestations", "attestation_id", attestation_id)

    def record_meta(self, record_id: str) -> sqlite3.Row:
        connection = self._connect()
        try:
            row = connection.execute("SELECT * FROM records WHERE record_id=?", (record_id,)).fetchone()
            if row is None:
                raise StorageError(f"missing record {record_id}")
            return row
        finally:
            connection.close()

    def dependencies(self, record_id: str) -> tuple[str, ...]:
        return self._relation("record_dependencies", "record_id", "dependency_id", record_id)

    def findings_for(self, record_id: str) -> tuple[str, ...]:
        return self._relation("record_findings", "record_id", "finding_id", record_id)

    def _relation(self, table: str, left: str, right: str, value: str) -> tuple[str, ...]:
        connection = self._connect()
        try:
            return tuple(row[0] for row in connection.execute(f"SELECT {right} FROM {table} WHERE {left}=? ORDER BY {right}", (value,)))
        finally:
            connection.close()

    def indexed_ids(self, semantic_key: str) -> tuple[str, ...]:
        connection = self._connect()
        try:
            return tuple(row[0] for row in connection.execute("SELECT record_id FROM semantic_index WHERE semantic_key=? ORDER BY record_id", (semantic_key,)))
        finally:
            connection.close()

    def all_record_ids(self) -> tuple[str, ...]:
        connection = self._connect()
        try:
            return tuple(row[0] for row in connection.execute("SELECT record_id FROM records ORDER BY record_id"))
        finally:
            connection.close()

    def seal_revision(self, revision_id: str, snapshot: bytes, members: Mapping[str, bytes]) -> Path:
        """Write and verify one deterministic ZIP, then publish its DB row."""
        revision_id = _text(revision_id, "revision_id")
        if not isinstance(snapshot, bytes) or not isinstance(members, Mapping):
            raise StorageError("snapshot and members have invalid types")
        all_members = {"snapshot.json": snapshot, **dict(members)}
        if "manifest.json" in all_members or any(not name or name.startswith("/") or ".." in Path(name).parts for name in all_members):
            raise StorageError("invalid revision member path")
        if any(not isinstance(data, bytes) for data in all_members.values()):
            raise StorageError("revision members must be bytes")
        manifest = {"members": {name: _sha(data) for name, data in sorted(all_members.items(), key=lambda item: item[0].encode("utf-8"))}}
        manifest_bytes = canonical_json_bytes(manifest)
        all_members["manifest.json"] = manifest_bytes
        filename = f"{revision_id}.zip"
        destination = self.revisions_dir / filename
        temporary = self.revisions_dir / f"{revision_id}.zip.tmp"
        if destination.exists():
            if destination.read_bytes() != self._read_bytes(temporary if temporary.exists() else destination):
                self.verify_zip(destination, revision_id=revision_id)
            self.verify_zip(destination, revision_id=revision_id)
        else:
            try:
                with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
                    # The manifest is deliberately the final member.  All
                    # closure paths before it are UTF-8 lexical sorted.
                    member_names = sorted((name for name in all_members if name != "manifest.json"),
                                          key=lambda item: item.encode("utf-8")) + ["manifest.json"]
                    for name in member_names:
                        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                        info.create_system = 3
                        info.create_version = 20
                        info.extract_version = 20
                        info.external_attr = 0o100444 << 16
                        info.extra = b""
                        info.comment = b""
                        info.flag_bits = 0
                        info.compress_type = zipfile.ZIP_STORED
                        archive.writestr(info, all_members[name])
                    archive.fp.flush()
                    os.fsync(archive.fp.fileno())
                self.verify_zip(temporary, revision_id=revision_id)
                os.replace(temporary, destination)
                _fsync_dir(self.revisions_dir)
            finally:
                if temporary.exists():
                    temporary.unlink()
        zip_sha = _sha(destination.read_bytes())
        snapshot_json = snapshot
        def operation(connection: sqlite3.Connection) -> None:
            row = connection.execute("SELECT * FROM revisions WHERE revision_id=?", (revision_id,)).fetchone()
            values = (revision_id, filename, zip_sha, snapshot_json)
            if row is not None and tuple(row) != values:
                raise StorageError("immutable revision ID has conflicting content")
            if row is None:
                connection.execute("INSERT INTO revisions VALUES(?,?,?,?)", values)
        self._write(operation)
        return destination

    @staticmethod
    def _read_bytes(path: Path) -> bytes:
        return path.read_bytes()

    def verify_zip(self, path: Path | str, *, revision_id: str | None = None) -> Mapping[str, bytes]:
        path = Path(path)
        try:
            with zipfile.ZipFile(path, "r") as archive:
                infos = archive.infolist()
                names = [info.filename for info in infos]
                if (len(names) != len(set(names)) or names[-1:] != ["manifest.json"]
                        or names[:-1] != sorted(names[:-1], key=lambda n: n.encode("utf-8"))):
                    raise StorageError("revision ZIP members are not unique and UTF-8 sorted")
                for info in infos:
                    expected = zipfile.ZipInfo(info.filename, date_time=(1980, 1, 1, 0, 0, 0))
                    if (info.date_time != expected.date_time or info.create_system != 3 or info.create_version != 20
                            or info.extract_version != 20 or info.external_attr != (0o100444 << 16)
                            or info.extra != b"" or info.comment != b"" or info.flag_bits != 0
                            or info.compress_type != zipfile.ZIP_STORED):
                        raise StorageError("revision ZIP metadata is not deterministic")
                if names[-1:] != ["manifest.json"]:
                    raise StorageError("revision ZIP has no final manifest")
                for name in names[:-1]:
                    parts = name.split("/")
                    if (name == "snapshot.json" or len(parts) != 2 or parts[0] not in {"records", "findings", "edges", "attestations"}
                            or not parts[1].endswith(".json") or not parts[1][:-5]):
                        if name != "snapshot.json":
                            raise StorageError("unexpected revision ZIP member")
                values = {name: archive.read(name) for name in names}
        except (OSError, zipfile.BadZipFile, IndexError) as exc:
            raise StorageError("invalid revision ZIP") from exc
        try:
            manifest = json.loads(values["manifest.json"].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StorageError("invalid revision manifest") from exc
        if canonical_json_bytes(manifest) != values["manifest.json"] or set(manifest) != {"members"} or not isinstance(manifest["members"], dict):
            raise StorageError("non-canonical revision manifest")
        expected_names = set(manifest["members"]) | {"manifest.json"}
        if set(values) != expected_names or "snapshot.json" not in manifest["members"]:
            raise StorageError("revision manifest member set mismatch")
        for name, digest in manifest["members"].items():
            if not isinstance(name, str) or not isinstance(digest, str) or name == "manifest.json" or name not in values:
                raise StorageError("revision manifest member is invalid")
            try:
                parsed = json.loads(values[name].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise StorageError("revision member is not canonical JSON") from exc
            if canonical_json_bytes(parsed) != values[name] or _sha(values[name]) != digest:
                raise StorageError("revision manifest hash mismatch")
        if revision_id is not None:
            try:
                snapshot = json.loads(values["snapshot.json"].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise StorageError("invalid revision snapshot") from exc
            if not isinstance(snapshot, dict) or snapshot.get("revision_id") != revision_id:
                raise StorageError("revision snapshot ID mismatch")
        return values

    def register_revision(self, revision_id: str, zip_filename: str, zip_sha256: str, snapshot: bytes) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            row = connection.execute("SELECT * FROM revisions WHERE revision_id=?", (revision_id,)).fetchone()
            values = (revision_id, zip_filename, zip_sha256, snapshot)
            if row is not None and tuple(row) != values:
                raise StorageError("immutable revision row conflict")
            if row is None:
                connection.execute("INSERT INTO revisions VALUES(?,?,?,?)", values)
        self._write(operation)

    def activate(self, revision_id: str) -> None:
        revision_id = _text(revision_id, "revision_id")
        self._write(lambda c: c.execute("INSERT INTO active_revision(slot, revision_id) VALUES(1,?) ON CONFLICT(slot) DO UPDATE SET revision_id=excluded.revision_id", (revision_id,)))

    def active_revision_id(self) -> str | None:
        connection = self._connect()
        try:
            row = connection.execute("SELECT revision_id FROM active_revision WHERE slot=1").fetchone()
            return None if row is None else row[0]
        finally:
            connection.close()

    def revision_row(self, revision_id: str) -> sqlite3.Row:
        connection = self._connect()
        try:
            row = connection.execute("SELECT * FROM revisions WHERE revision_id=?", (revision_id,)).fetchone()
            if row is None:
                raise StorageError(f"missing revision {revision_id}")
            return row
        finally:
            connection.close()


# Explicit name for callers that prefer the implementation's storage role.
SQLiteStorage = Storage
V2Storage = Storage

__all__ = ["DB_NAME", "Storage", "SQLiteStorage", "V2Storage", "StorageError"]
