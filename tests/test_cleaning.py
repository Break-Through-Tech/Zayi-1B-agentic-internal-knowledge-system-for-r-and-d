from src.cleaning import (
    CLEANER_VERSION,
    build_vocab,
    clean_pages,
    clean_text,
)


def test_expands_ligatures():
    assert clean_text("eﬃcient ﬁne tuning") == "efficient fine tuning"


def test_removes_control_chars_without_fusing_words():
    assert clean_text("loss\x03function") == "loss function"


def test_maps_cp1252_strays():
    assert clean_text("range 1\x9610") == "range 1–10"
    assert clean_text("\x95 item") == "• item"


def test_joins_simple_line_wraps_by_default():
    assert clean_text("atten-\ntion is all") == "attention is all"


def test_keeps_hyphen_when_corpus_shows_compound_inline():
    # "fine-tuning" appears unbroken elsewhere in the corpus -> hyphen is real
    vocab = build_vocab(["we study fine-tuning and cross-encoder models"])
    assert clean_text("fine-\ntuning", vocab) == "fine-tuning"
    assert clean_text("cross-\nencoder", vocab) == "cross-encoder"


def test_fuses_when_corpus_shows_fused_word_inline():
    vocab = build_vocab(["the attention mechanism"])
    assert clean_text("atten-\ntion", vocab) == "attention"


def test_multi_hyphen_compounds_keep_hyphen_even_without_evidence():
    # regression: used to produce "state-of-theart"
    assert clean_text("state-of-the-\nart results") == "state-of-the-art results"
    assert clean_text("knowledge-\nintensive", build_vocab(["knowledge-intensive tasks"])) == "knowledge-intensive"


def test_keeps_capitalized_breaks_and_inline_hyphens():
    # A hyphen before a capitalized line (e.g. a name) is not a wrap artifact
    assert clean_text("Wilcoxon-\nMann test") == "Wilcoxon-\nMann test"
    assert clean_text("state-of-the-art") == "state-of-the-art"


def test_collapses_whitespace_but_keeps_newlines():
    assert clean_text("a  b\t c \n d") == "a b c\nd"


def test_idempotent():
    vocab = build_vocab(["fine-tuning attention"])
    text = "eﬃcient atten-\ntion and fine-\ntuning  \x03 test"
    once = clean_text(text, vocab)
    assert clean_text(once, vocab) == once


def test_clean_pages_uses_whole_corpus_as_evidence():
    # evidence for "fine-tuning" is on a DIFFERENT page (and ligatured)
    pages = [
        {"paper_id": "a", "page_number": 1, "text": "we discuss ﬁne-tuning here"},
        {"paper_id": "a", "page_number": 2, "text": "results of ﬁne-\ntuning improve"},
    ]
    cleaned = clean_pages(pages)
    assert cleaned[1]["text"] == "results of fine-tuning improve"


def test_clean_pages_preserves_metadata_and_tags_version():
    pages = [{"paper_id": "x", "page_number": 3, "text": "ﬁne"}]
    (cleaned,) = clean_pages(pages)
    assert cleaned["text"] == "fine"
    assert cleaned["paper_id"] == "x"
    assert cleaned["page_number"] == 3
    assert cleaned["cleaner"] == CLEANER_VERSION
    assert pages[0]["text"] == "ﬁne"  # input not mutated
