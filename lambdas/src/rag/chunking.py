def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Fixed-size character windows with overlap (same scheme as the R1-R8 runs)."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return [c for c in chunks if c.strip()]
