from src.cleaning import CLEANER_VERSION, clean_pages, clean_text


def test_expands_ligatures():
    assert clean_text("eﬃcient ﬁne-tuning") == "efficient fine-tuning"


def test_removes_control_chars_without_fusing_words():
    assert clean_text("loss\x03function") == "loss function"


def test_maps_cp1252_strays():
    assert clean_text("range 1\x9610") == "range 1–10"
    assert clean_text("\x95 item") == "• item"


def test_joins_hyphenated_line_wraps():
    assert clean_text("atten-\ntion is all") == "attention is all"


def test_keeps_real_hyphens_and_capitalized_breaks():
    # A hyphen before a capitalized line (e.g. a name) is not a wrap artifact
    assert clean_text("Wilcoxon-\nMann test") == "Wilcoxon-\nMann test"
    assert clean_text("state-of-the-art") == "state-of-the-art"


def test_collapses_whitespace_but_keeps_newlines():
    assert clean_text("a  b\t c \n d") == "a b c\nd"


def test_idempotent():
    text = "eﬃcient atten-\ntion  \x03 test"
    once = clean_text(text)
    assert clean_text(once) == once


def test_clean_pages_preserves_metadata_and_tags_version():
    pages = [{"paper_id": "x", "page_number": 3, "text": "ﬁne"}]
    (cleaned,) = clean_pages(pages)
    assert cleaned["text"] == "fine"
    assert cleaned["paper_id"] == "x"
    assert cleaned["page_number"] == 3
    assert cleaned["cleaner"] == CLEANER_VERSION
    assert pages[0]["text"] == "ﬁne"  # input not mutated
