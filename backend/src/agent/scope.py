"""Retrieval scope helpers shared by chat, exam, dense search, and BM25."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal


ScopeType = Literal["all", "project", "folder", "files", "week", "doc_type"]


@dataclass(frozen=True)
class RetrievalScope:
    type: ScopeType = "all"
    project_id: str | None = None
    folder_id: str | None = None
    file_ids: tuple[str, ...] = field(default_factory=tuple)
    week: str | None = None
    doc_types: tuple[str, ...] = field(default_factory=tuple)
    explicit: bool = False

    @classmethod
    def from_input(
        cls,
        scope: dict[str, Any] | None = None,
        week_filter: str | None = None,
    ) -> "RetrievalScope":
        if not scope:
            if week_filter:
                return cls(type="week", week=week_filter, explicit=True)
            return cls()
        scope_type = scope.get("type") or "all"
        if scope_type not in {"all", "project", "folder", "files", "week", "doc_type"}:
            scope_type = "all"
        file_ids = tuple(fid for fid in (scope.get("file_ids") or []) if fid)
        doc_types = tuple(dt for dt in (scope.get("doc_types") or []) if dt)
        return cls(
            type=scope_type,
            project_id=scope.get("project_id"),
            folder_id=scope.get("folder_id"),
            file_ids=file_ids,
            week=scope.get("week") or week_filter,
            doc_types=doc_types,
            explicit=True,
        )

    def is_explicit_empty(self) -> bool:
        return self.explicit and self.type == "files" and not self.file_ids

    def metadata_filter(self) -> dict[str, Any]:
        where: dict[str, Any] = {}
        if self.project_id:
            where["project_id"] = self.project_id
        if self.folder_id:
            where["folder_id"] = self.folder_id
        if self.file_ids:
            where["file_id"] = {"$in": list(self.file_ids)}
        if self.week:
            where["week"] = self.week
        if self.doc_types:
            where["type"] = {"$in": list(self.doc_types)}
        return where

    def snapshot(self) -> dict[str, Any] | None:
        if not self.explicit and self.type == "all":
            return None
        out: dict[str, Any] = {"type": self.type}
        if self.project_id:
            out["project_id"] = self.project_id
        if self.folder_id:
            out["folder_id"] = self.folder_id
        if self.file_ids or self.type == "files":
            out["file_ids"] = list(self.file_ids)
        if self.week:
            out["week"] = self.week
        if self.doc_types:
            out["doc_types"] = list(self.doc_types)
        return out

    def scope_hash(self) -> str:
        payload = self.snapshot() or {"type": "all"}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]
