"""
src/rag/chunking.py
─────────────────────────────────────────────────────────────────────
Divide laudos de checagem em chunks com sobreposição e gera os
metadados/IDs determinísticos usados na indexação no Qdrant.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, DEFAULT_LANGUAGE

_ID_NAMESPACE = uuid.UUID("6f1c2f0e-3b7a-4c55-9a43-0d6f6f3b2a10")
_SENTENCE_END = re.compile(r"[.!?…](?=\s)")


@dataclass(frozen=True)
class Chunk:
    document_id: str
    chunk_id: str
    chunk_index: int
    text: str
    titulo: str = ""
    url: str = ""
    dominio: str = ""
    data_publicacao: str = ""
    idioma: str = DEFAULT_LANGUAGE
    content_hash: str = ""
    extra_payload: dict = field(default_factory=dict)

    @property
    def embed_text(self) -> str:
        """Texto usado no embedding: título + chunk, para manter contexto."""
        return f"{self.titulo}. {self.text}" if self.titulo else self.text

    def payload(self) -> dict:
        payload = {
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "texto_chunk": self.text,
            "titulo": self.titulo,
            "url": self.url,
            "dominio": self.dominio,
            "data_publicacao": self.data_publicacao,
            "idioma": self.idioma,
            "content_hash": self.content_hash,
        }
        payload.update(self.extra_payload)
        return payload


def document_id_for(url: str) -> str:
    """ID estável do documento, derivado da URL."""
    return hashlib.sha1(url.strip().encode("utf-8")).hexdigest()


def point_id_for(document_id: str, chunk_index: int) -> str:
    """UUID determinístico (aceito pelo Qdrant) para um chunk."""
    return str(uuid.uuid5(_ID_NAMESPACE, f"{document_id}:{chunk_index}"))


def _find_cut(text: str, start: int, hard_end: int) -> int:
    """Melhor ponto de corte em (start, hard_end]: fim de frase, senão espaço."""
    min_end = start + (hard_end - start) // 2
    window = text[min_end:hard_end]
    ends = [m.end() for m in _SENTENCE_END.finditer(window)]
    if ends:
        return min_end + ends[-1]
    space = text.rfind(" ", min_end, hard_end)
    return space if space > start else hard_end


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """Divide `text` em chunks de até `chunk_size` chars com `overlap` de sobreposição."""
    if chunk_size <= 0:
        raise ValueError("chunk_size deve ser positivo")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap deve estar em [0, chunk_size)")

    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        hard_end = min(start + chunk_size, n)
        end = n if hard_end == n else _find_cut(text, start, hard_end)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        # Recua `overlap` chars, alinhando ao início de palavra, e garante progresso.
        next_start = end - overlap
        if overlap:
            space = text.find(" ", next_start, end)
            if space != -1:
                next_start = space + 1
        start = max(next_start, start + 1)
    return chunks


def build_chunks(
    record: dict,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Gera os `Chunk`s de um registro (`titulo`, `texto`, `url`, ...) do JSONL."""
    url = (record.get("url") or "").strip()
    texto = record.get("texto") or ""
    document_id = document_id_for(url)
    content_hash = hashlib.sha256(
        f"{record.get('titulo') or ''}\n{texto}".encode("utf-8")
    ).hexdigest()
    extra_payload = {
        key: record[key]
        for key in (
            "layer",
            "preprocessing_version",
            "normalized_text_hash",
            "duplicate_group_id",
            "is_duplicate",
            "duplicate_of_url",
            "injection_risk",
            "injection_matches",
            "topic",
            "topic_score",
            "topic_keywords",
            "entity_candidates",
            "document_quality_score",
        )
        if key in record
    }

    return [
        Chunk(
            document_id=document_id,
            chunk_id=point_id_for(document_id, i),
            chunk_index=i,
            text=piece,
            titulo=(record.get("titulo") or "").strip(),
            url=url,
            dominio=record.get("dominio") or "",
            data_publicacao=record.get("data_publicacao") or "",
            idioma=record.get("idioma") or DEFAULT_LANGUAGE,
            content_hash=content_hash,
            extra_payload=extra_payload,
        )
        for i, piece in enumerate(chunk_text(texto, chunk_size, overlap))
    ]
