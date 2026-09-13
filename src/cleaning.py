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

CLEANER_VERSION = "seed-v2"

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

# A lowercase word (possibly itself hyphenated, like "state-of-the") broken
# across a line by a hyphen. Capitalized right-hand sides (names, "Wilcoxon-
# Mann") are left alone.
_HYPHEN_WRAP_RE = re.compile(r"([a-z][\w-]*)-\n([a-z]\w*)")

_WORD_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")

_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_SPACE_AROUND_NEWLINE_RE = re.compile(r" ?\n ?")


def build_vocab(texts):
    """Lowercased words (plain and hyphenated) seen anywhere in the corpus.
    Used as evidence for resolving hyphenated line-wraps."""
    vocab = set()
    for text in texts:
        for word in _WORD_RE.findall(text):
            vocab.add(word.lower())
    return frozenset(vocab)


def _join_wrap(match, vocab):
    """A line-wrap hyphen is only removed when the corpus itself shows the
    fused word exists; if the corpus instead shows the hyphenated compound
    inline (e.g. "fine-tuning", "state-of-the-art"), the hyphen is real and
    stays. Compounds that already contain a hyphen default to keeping it."""
    left, right = match.group(1), match.group(2)
    hyphenated = f"{left}-{right}".lower()
    fused = f"{left}{right}".lower()
    if hyphenated in vocab:
        return f"{left}-{right}"
    if fused in vocab or "-" not in left:
        return f"{left}{right}"
    return f"{left}-{right}"


def clean_text(text, vocab=frozenset()):
    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)
    for src, dst in _CP1252.items():
        text = text.replace(src, dst)
    text = _CONTROL_RE.sub(" ", text)
    text = _HYPHEN_WRAP_RE.sub(lambda m: _join_wrap(m, vocab), text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _SPACE_AROUND_NEWLINE_RE.sub("\n", text)
    return text.strip()


def clean_pages(pages):
    """Return new page records with cleaned text, tagged with the cleaner
    version so chunks/embeddings can be traced to the cleaning pass that
    produced them. Line-wrap hyphens are resolved with evidence from the
    whole corpus (see _join_wrap), so cleaning is done corpus-at-a-time."""
    # De-ligature before collecting vocabulary so "ﬁne-tuning" in running
    # text counts as evidence for "fine-tuning" at a line wrap.
    deligatured = []
    for page in pages:
        text = page["text"]
        for src, dst in _LIGATURES.items():
            text = text.replace(src, dst)
        deligatured.append(text)
    vocab = build_vocab(deligatured)

    cleaned = []
    for page in pages:
        record = dict(page)
        record["text"] = clean_text(page["text"], vocab)
        record["cleaner"] = CLEANER_VERSION
        cleaned.append(record)
    return cleaned
