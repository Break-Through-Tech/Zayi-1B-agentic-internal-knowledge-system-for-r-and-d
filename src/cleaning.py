"""Text cleaning for the extracted paper pages.

Interim cleaning pass so chunking (Task 4) and the vector DB (Task 5) aren't
blocked on Task 3. When the Task 3 preprocessing lands, its output can replace
`clean_pages` as long as it returns the same page-record shape — everything
downstream only sees (text, metadata) records.

Rules are targeted at artifacts actually measured in the dataset:
ligatures (~690), C0 control characters (~2,400), Windows-1252 mojibake
(bullets/dashes), and hyphenated line-wraps (~1,100).
"""

import re

CLEANER_VERSION = "seed-v1"

# Ligatures as extracted from the PDFs
_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
}

# Stray Windows-1252 bytes that survived extraction as C1 codepoints
_CP1252 = {
    "\x91": "'",
    "\x92": "'",
    "\x93": '"',
    "\x94": '"',
    "\x95": "•",  # bullet
    "\x96": "–",  # en dash
    "\x97": "—",  # em dash
}

# C0 control characters except \n and \t. These are mostly failed glyph
# mappings (math symbols); replaced with a space so words don't fuse.
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")

# "atten-\ntion" -> "attention" (lowercase letters on both sides only, so
# hyphenated names/ranges split across lines aren't falsely joined)
_HYPHEN_WRAP_RE = re.compile(r"([a-z])-\n([a-z])")

_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_SPACE_AROUND_NEWLINE_RE = re.compile(r" ?\n ?")


def clean_text(text):
    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)
    for src, dst in _CP1252.items():
        text = text.replace(src, dst)
    text = _CONTROL_RE.sub(" ", text)
    text = _HYPHEN_WRAP_RE.sub(r"\1\2", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _SPACE_AROUND_NEWLINE_RE.sub("\n", text)
    return text.strip()


def clean_pages(pages):
    """Return new page records with cleaned text, tagged with the cleaner
    version so chunks/embeddings can be traced to the cleaning pass that
    produced them."""
    cleaned = []
    for page in pages:
        record = dict(page)
        record["text"] = clean_text(page["text"])
        record["cleaner"] = CLEANER_VERSION
        cleaned.append(record)
    return cleaned
