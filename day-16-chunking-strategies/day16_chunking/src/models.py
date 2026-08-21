from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class DocumentElement:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Chunk:
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    parent_id: Optional[str] = None
