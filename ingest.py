"""Ingest the Constitution of Pakistan PDF into a Chroma vector store.

Usage:
    python ingest.py constitution_of_Pakistan.pdf --reset
    python ingest.py constitution_of_Pakistan.pdf --dry-run    # no embedding model needed
"""
from __future__ import annotations

import argparse
import re
from bisect import bisect_right
from dataclasses import dataclass, field
from collections import Counter
import pymupdf




@dataclass
class ExtractorConfig:
    header_max_y: float = 80.0       # running header sits above this y
    body_x_min: float = 150.0        # body column left edge (left-margin notes are at x~91)
    body_x_max: float = 458.0        # body column right edge (right-margin notes are at x~463)
    footnote_font_max: float = 8.5   # footnotes are 8pt, body is 11pt
    marker_font_max: float = 7.0     # superscript footnote numbers are 4-5.5pt
    separator_min_len: int = 30      # footnote separator = long run of dashes
    note_y_tolerance: float = 6.0


@dataclass
class MarginNote:
    y0: float
    text: str


@dataclass
class PageContent:
    pdf_page: int
    printed_page: str | None
    body: str
    margin_notes: list[MarginNote] = field(default_factory=list)
    footnotes: list[tuple[str, str]] = field(default_factory=list)


_SEPARATOR_CHARS = set("–—-_ ")
_PAGE_TOKEN = re.compile(r"\d{1,3}|[ivxlc]{1,6}", re.I)
_FOOTER_ROMAN = re.compile(r"^\(([ivxlc]+)\)$", re.I)
KEEP_HYPHEN = {"co-operative", "co-ordination", "up-to", "re-numbered"}
_WORD = re.compile(r"(?<![-\w])[A-Za-z]{3,}(?![-\w])")
_SPLIT = re.compile(r"\b([A-Za-z]{2,})-([A-Za-z]{2,})\b")

def build_vocab(full_text: str) -> set[str]:
    """Standalone words (not touching a hyphen) from the WHOLE document."""
    return {w.lower() for w in _WORD.findall(full_text)}

def fix_hyphens(text: str, vocab: set[str], log: Counter | None = None) -> str:
    def repl(m):
        a, b = m.group(1), m.group(2)
        pair = f"{a}-{b}".lower()
        if pair in KEEP_HYPHEN or (a + b).lower() not in vocab:
            return m.group(0)
        if log is not None:
            log[pair] += 1
        return a + b
    return _SPLIT.sub(repl, text)


class PDFExtractor:
    def __init__(self, config: ExtractorConfig | None = None):
        self.cfg = config or ExtractorConfig()

    def extract(self, path: str) -> list[PageContent]:
        doc = pymupdf.open(path)
        pages = [self._extract_page(p, i + 1) for i, p in enumerate(doc)]
        doc.close()
        return pages

    def _extract_page(self, page, pdf_page: int) -> PageContent:
        c = self.cfg
        blocks = [b for b in page.get_text("dict")["blocks"] if b["type"] == 0]
        sep_y = self._find_separator_y(blocks)

        printed_page = None
        rows = []             # (baseline_y, x0, text)
        margin_spans = []     # (side, baseline, text), one per visual margin line
        footnote_lines = []   # each line = [(font_size, text), ...]

        for b in blocks:
            # header: running title + page number
            if b["bbox"][1] < c.header_max_y:
                big = " ".join(s["text"] for ln in b["lines"] for s in ln["spans"]
                               if s["size"] > c.marker_font_max)
                big = re.sub(r"\s+", " ", big).strip()
                rest = re.sub(r"CONSTITUTION OF PAKISTAN", "", big, flags=re.I).strip()
                if "CONSTITUTION OF PAKISTAN" in big.upper() or _PAGE_TOKEN.fullmatch(rest):
                    if _PAGE_TOKEN.fullmatch(rest):
                        printed_page = rest
                    continue

            for ln in b["lines"]:
                # classify each span on its own: a margin title can share a row with body text
                body_sp, margin_sp = [], []
                for s_ in ln["spans"]:
                    x = s_["bbox"][0]
                    if s_["text"].strip() == "":
                        body_sp.append(s_)
                    elif x < c.body_x_min or x > c.body_x_max:
                        margin_sp.append(s_)
                    else:
                        body_sp.append(s_)

                m_txt = "".join(s_["text"] for s_ in margin_sp
                                if s_["size"] > c.marker_font_max).strip()
                if m_txt:
                    side = "L" if margin_sp[0]["bbox"][0] < c.body_x_min else "R"
                    base = max(margin_sp, key=lambda q: q["size"])["origin"][1]
                    margin_spans.append((side, base, m_txt))

                if not any(s_["text"].strip() for s_ in body_sp):
                    continue
                if sep_y is not None and ln["bbox"][1] >= sep_y:
                    footnote_lines.append([(s_["size"], s_["text"]) for s_ in body_sp])
                else:
                    txt = "".join(s_["text"] for s_ in body_sp if s_["size"] > c.marker_font_max)
                    if txt.strip():
                        base = max(body_sp, key=lambda q: q["size"])["origin"][1]
                        rows.append((base, body_sp[0]["bbox"][0], txt.rstrip()))

        margin_notes = self._group_margin_lines(margin_spans)
        footnotes = self._build_footnotes(footnote_lines)
        rows = self._merge_same_row(rows)

        if printed_page is None:  # roman numeral footer like "(xv)"
            for _, _, text in rows:
                m = _FOOTER_ROMAN.match(text.strip())
                if m:
                    printed_page = m.group(1)
                    break

        margin_notes.sort(key=lambda n: n.y0)
        body = self._assemble_body(rows, margin_notes)
        return PageContent(pdf_page, printed_page, body, margin_notes, footnotes)
    
    @staticmethod
    def _group_margin_lines(lines) -> list[MarginNote]:
        """Join consecutive margin lines (same side, <=12pt apart) into one note."""
        notes = []
        for side in ("L", "R"):
            cur, last = None, None
            for _, base, text in sorted((l for l in lines if l[0] == side), key=lambda l: l[1]):
                if cur is not None and base - last <= 12:
                    cur[1] += " " + text
                else:
                    if cur:
                        notes.append(cur)
                    cur = [base, text]
                last = base
            if cur:
                notes.append(cur)
        return [MarginNote(b, re.sub(r"\s+", " ", t).strip())
                for b, t in notes if re.search(r"[A-Za-z]", t)]
    
    @staticmethod
    def _baseline(line) -> float:
        # baseline of the largest span: stable even when a superscript changes the bbox
        return max(line["spans"], key=lambda s: s["size"])["origin"][1]

    @staticmethod
    def _merge_same_row(rows):
        rows = sorted(rows, key=lambda r: (round(r[0], 0), r[1]))
        merged = []
        for base, x0, text in rows:
            if merged and abs(merged[-1][0] - base) < 2.0:
                pb, px, pt = merged[-1]
                merged[-1] = (pb, px, pt + " " + text.strip())
            else:
                merged.append((base, x0, text))
        return merged

    def _assemble_body(self, rows, notes) -> str:
        if not rows:
            return ""
        gaps = sorted(rows[i][0] - rows[i - 1][0] for i in range(1, len(rows)))
        pitch = gaps[len(gaps) // 4] if gaps else 13.0
        para_gap = max(pitch * 1.35, 1.0)

        # attach each margin note to the nearest row at/above it
        note_at = {}
        for note in notes:
            idx = 0
            for i, (base, _, _) in enumerate(rows):
                if base <= note.y0 + self.cfg.note_y_tolerance:
                    idx = i
            note_at.setdefault(idx, []).append(note.text)

        out = []
        for i, (base, _, text) in enumerate(rows):
            if i in note_at:
                out.append("\n\n" if out else "")
                out.append("\n".join(f"[[NOTE: {t}]]" for t in note_at[i]) + "\n")
            elif out:
                out.append("\n\n" if base - rows[i - 1][0] > para_gap else "\n")
            out.append(text)
        return "".join(out)

    def _find_separator_y(self, blocks):
        c = self.cfg
        for b in blocks:
            for ln in b["lines"]:
                text = "".join(s["text"] for s in ln["spans"]).strip()
                size = max((s["size"] for s in ln["spans"]), default=99)
                if (len(text) >= c.separator_min_len and set(text) <= _SEPARATOR_CHARS
                        and size <= c.footnote_font_max):
                    return ln["bbox"][1]
        return None

    def _build_footnotes(self, lines):
        c = self.cfg
        notes = []  # [number, text]
        for spans in lines:
            buf = ""
            for size, txt in spans:
                if size <= c.marker_font_max and txt.strip():
                    if buf.strip() and notes:
                        notes[-1][1] += " " + buf.strip()
                    buf = ""
                    notes.append([txt.strip().rstrip("."), ""])
                else:
                    buf += txt
            if buf.strip() and notes:
                notes[-1][1] += " " + buf.strip()
        return [(n, re.sub(r"\s+", " ", t).strip()) for n, t in notes if t.strip()]




@dataclass
class CleanedDocument:
    text: str
    page_starts: list         # (char_offset, pdf_page, printed_page)
    footnotes_by_page: dict   # pdf_page -> ["1. Subs. by ...", ...]

    def page_at(self, offset: int):
        i = bisect_right([s[0] for s in self.page_starts], offset) - 1
        _, pdf_page, printed = self.page_starts[max(i, 0)]
        return pdf_page, printed


class TextCleaner:
    TERMINAL = ".;:,–—?!"
    LABEL = re.compile(r"^(\(\w{1,5}\)|\d{1,3}[A-Z]{0,2}\.)(\s|$)")
    HEADING = re.compile(r"^(PART [IVXLC]+\b|CHAPTER \d+[A-Z]?\b)")
    XREF_FOOTNOTE = re.compile(r"^See\b.*\b(footnote|supra)\b", re.I)
    NOTE = re.compile(r"\[\[NOTE: (.*?)\]\]")

    def __init__(self, first_content_page: int = 23):
        self.first_content_page = first_content_page
        self.hyphen_log = Counter()

    def clean(self, pages) -> CleanedDocument:
        # pass 1: clean every page, as before
        cleaned = []
        for p in pages:
            if p.pdf_page < self.first_content_page:
                continue
            body = self._clean_page(p.body)
            if not body:
                continue
            notes = [f"{n}. {t}" for n, t in p.footnotes if not self.XREF_FOOTNOTE.match(t)]
            cleaned.append((p, body, notes))

        # vocabulary from the whole document (bodies + footnotes)
        everything = "\n".join(b for _, b, _ in cleaned) + "\n" + \
                    "\n".join(n for _, _, ns in cleaned for n in ns)
        vocab = build_vocab(everything)
        self.hyphen_log = Counter()

        # pass 2: fix hyphens per page, then assemble so offsets stay correct
        text, starts, footnotes = "", [], {}
        for p, body, notes in cleaned:
            body = fix_hyphens(body, vocab, self.hyphen_log)
            if text:
                text += self._joiner(text, body)
            starts.append((len(text), p.pdf_page, p.printed_page))
            text += body
            if notes:
                footnotes[p.pdf_page] = [fix_hyphens(n, vocab, self.hyphen_log) for n in notes]
        return CleanedDocument(text, starts, footnotes)    

    # ---------------------------------------------------------------- per page
    def _clean_page(self, body: str) -> str:
        if not body.strip():
            return ""
        # 1) protect margin notes; each becomes its own paragraph
        stash = []
        def _stash(m):
            stash.append(self._tidy(m.group(1)))
            return f"\x01{len(stash) - 1}\x02\n\n"
        t = self.NOTE.sub(_stash, body)

        # 2) omitted provisions ("46. *", "(2) * * *") -> stub; other asterisks = repealed words
        t = re.sub(r"(?m)^(\(\w{1,5}\)|\d{1,3}[A-Z]{0,2}\.)[^\S\n]*[\*\s]+$", r"\1 Omitted.", t)
        t = re.sub(r"(?m)^[\*\s]+$", "", t)
        t = re.sub(r"[^\S\n]*(?:\*[^\S\n]*)+", " ", t)

        # 3) square brackets mark amended text: drop them, tidy spacing
        t = self._tidy(t)

        # 4) rebuild paragraphs from wrapped lines
        paragraphs = []
        for para in re.split(r"\n\s*\n", t):
            lines = [re.sub(r"[^\S\n]+", " ", ln).strip() for ln in para.split("\n")]
            lines = [self._norm_heading(ln) for ln in lines if ln]
            for group in self._split_at_headings(lines):
                paragraphs.append(self._join_lines(group))
        paragraphs = self._fix_headings_and_notes(paragraphs)
        out = "\n\n".join(paragraphs)

        # 5) restore margin notes
        return re.sub(r"\x01(\d+)\x02",
                      lambda m: f"[[NOTE: {stash[int(m.group(1))]}]]", out).strip()

    @staticmethod
    def _tidy(s: str) -> str:
        s = s.replace("[", " ").replace("]", " ")
        s = s.replace("‘‘", "“").replace("’’", "”")
        s = re.sub(r"(?<=[A-Za-z])- (?=[a-z])", "-", s)       # "well- being" -> "well-being"
        s = re.sub(r"[^\S\n]+([;,.:)])", r"\1", s)             # "Minister ;" -> "Minister;"
        s = re.sub(r"(?<!\.)\.\.(?!\.)", ".", s)               # stray ".."
        s = re.sub(r"“[^\S\n]+", "“", s)
        s = re.sub(r"[^\S\n]+”", "”", s)
        return re.sub(r"[^\S\n]+", " ", s).strip(" ") if "\n" not in s else s

    def _fix_headings_and_notes(self, paras):
        """PART X + title on one line; keep margin notes by their article, not above a heading."""
        out, i = [], 0
        while i < len(paras):
            p = paras[i]
            m = re.match(r"^PART ([IVXLC]+)\b[^\S\n]*[–—-]*[^\S\n]*(.*)$", p)
            if m and "\n" not in p:
                title, held = m.group(2).strip(), []
                if not title:
                    k = i + 1
                    while k < len(paras) and paras[k].startswith("\x01"):  # notes before the title
                        held.append(paras[k]); k += 1
                    if k < len(paras):
                        nxt = paras[k]
                        if (len(nxt) <= 80 and not nxt.endswith(".") and not nxt.startswith("(")
                                and not self.HEADING.match(nxt) and not re.match(r"^\d", nxt)):
                            title, i = nxt, k
                        else:
                            held = []
                out.append(f"PART {m.group(1)} – {title}" if title else f"PART {m.group(1)}")
                out.extend(held)
            else:
                out.append(p)
            i += 1
        # a note directly above a PART/CHAPTER heading belongs after the heading(s)
        j = 0
        while j < len(out) - 1:
            if out[j].startswith("\x01") and self.HEADING.match(out[j + 1]):
                out[j], out[j + 1] = out[j + 1], out[j]
                j = max(j - 1, 0)
            else:
                j += 1
        return out

    ARTICLE_LINE = re.compile(r"^\d{1,3}[A-Z]{0,2}\.\s")

    def _split_at_headings(self, lines):
        """A PART/CHAPTER line always starts a new paragraph; so does an article right after a CHAPTER line."""
        groups, cur = [], []
        for ln in lines:
            if cur and (self.HEADING.match(ln)
                        or (cur[-1].startswith("CHAPTER") and self.ARTICLE_LINE.match(ln))):
                groups.append(cur)
                cur = []
            cur.append(ln)
        if cur:
            groups.append(cur)
        return groups
    
    @staticmethod
    def _norm_heading(line: str) -> str:
        m = re.match(r"^CHAPTER\s+(\d+[A-Z]?)\s*\.?\s*[–—-]+\s*(.*)$", line)
        return f"CHAPTER {m.group(1)} – {m.group(2)}".strip() if m else line

    def _join_lines(self, lines):
        out = lines[0]
        for ln in lines[1:]:
            if re.search(r"\w-$", out):                              # wrap inside hyphenated word
                out += ln
            elif self.LABEL.match(ln) and out[-1] in self.TERMINAL:  # new clause starts here
                out += "\n" + ln
            else:
                out += " " + ln
        return out

    def _joiner(self, prev: str, nxt: str) -> str:
        """Glue page N to page N+1: continue the sentence, or start a new paragraph."""
        ends_sentence = prev.rstrip()[-1:] in self.TERMINAL
        starts_new = nxt.startswith("[[NOTE:") or self.HEADING.match(nxt)
        return "\n\n" if (ends_sentence or starts_new) else " "    



@dataclass
class Chunk:
    id: str
    text: str        # what gets embedded: context header + body
    metadata: dict   # flat values only (str/int/float/bool): Chroma rejects lists and None


class ArticleChunker:
    DOC = "Constitution of Pakistan"
    ART_START = re.compile(r"^(\d{1,3})([A-Z]{0,2})\.(?=[\s–—-])")
    PART = re.compile(r"^PART ([IVXLC]+)\b")
    CHAPTER = re.compile(r"^CHAPTER \d+[A-Z]?\b")
    SEPARATOR = re.compile(r"^[_–—\-\s]{3,}$")
    ANNEX_SPLIT = re.compile(r"(?m)^_{3,}\s*ANNEX")
    SCHEDULE = re.compile(r"^(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH) SCHEDULE\b")
    TERMINAL = ".;:,–—?!”)"

    def __init__(self, max_chars: int = 1500):
        self.max_chars = max_chars   # keeps header+body under bge-small's 512-token window

    # ------------------------------------------------------------------ public
    def chunk(self, doc) -> list[Chunk]:
        text = doc.text
        m = self.ANNEX_SPLIT.search(text)
        main, tail, tail_base = (text[:m.start()], text[m.start():], m.start()) if m else (text, "", 0)
        chunks = self._chunk_main(doc, main)
        if tail:
            chunks += self._chunk_tail(doc, tail, tail_base)
        return chunks

    # ----------------------------------------------- preamble + articles 1..280
    def _chunk_main(self, doc, text):
        chunks, buf = [], []                 # buf = preamble paragraphs
        part = chapter = sub = pending = None
        cur = None                           # article under construction
        last_key = None
        in_preamble = True

        paras = list(self._paras(text))
        for i, (off, p) in enumerate(paras):
            nxt = paras[i + 1][1] if i + 1 < len(paras) else ""
            if self.SEPARATOR.match(p):
                continue

            if self.PART.match(p):
                if in_preamble:
                    chunks += self._emit(doc, buf, "preamble", "Preamble", {"article_number": "Preamble"})
                    in_preamble = False
                chunks += self._flush(doc, cur); cur = None
                part, chapter, sub = p, None, None
                continue
            if in_preamble:
                buf.append((off, p)); continue

            if self.CHAPTER.match(p):
                chunks += self._flush(doc, cur); cur = None
                chapter, sub = p.rstrip("."), None
                continue

            if p.startswith("[[NOTE:"):
                note = p[len("[[NOTE: "):-2]
                if self._is_article_start(nxt, last_key) or self.CHAPTER.match(nxt) or self.PART.match(nxt):
                    pending = f"{pending} / {note}" if pending else note
                elif cur is not None and not cur["title"]:
                    cur["title"] = note            # note landed one line late: it titles the open article
                else:
                    pending = f"{pending} / {note}" if pending else note
                continue

            if self._is_article_start(p, last_key):
                chunks += self._flush(doc, cur)
                num, suf = self.ART_START.match(p).groups()
                last_key = (int(num), suf)
                cur = {"num": int(num), "suf": suf, "title": pending, "part": part,
                       "chapter": chapter, "sub": sub, "paras": [(off, p)]}
                pending = None
                continue

            # short unpunctuated paragraph right before a margin note = sub-heading inside a chapter
            if nxt.startswith("[[NOTE:") and len(p) < 120 and p[-1] not in self.TERMINAL and not p.startswith("("):
                chunks += self._flush(doc, cur); cur = None
                sub = p
                continue

            if cur is not None:
                cur["paras"].append((off, p))

        chunks += self._flush(doc, cur)
        return chunks

    def _is_article_start(self, p, last_key) -> bool:
        m = self.ART_START.match(p)
        if not m:
            return False
        key = (int(m.group(1)), m.group(2))
        return last_key is None or (key > last_key and key[0] - last_key[0] <= 10)

    def _flush(self, doc, a):
        if not a:
            return []
        num = f"{a['num']}{a['suf']}"
        extra = {"article_number": num, "article_num": a["num"]}
        # omitted articles carry their title inline: "71. Mediation Committee Omitted by ..."
        mo = re.match(r"^\d{1,3}[A-Z]{0,2}\.\s+(.*?)\.?\s+Omitted\b", a["paras"][0][1])
        if len(a["paras"]) == 1 and mo:
            extra["omitted"] = True
            a["title"] = a["title"] or mo.group(1).strip()
        for k_src, k_dst in (("title", "article_title"), ("part", "part"),
                             ("chapter", "chapter"), ("sub", "subheading")):
            if a[k_src]:
                extra[k_dst] = a[k_src]
        label = f"Article {num}" + (f": {a['title']}" if a["title"] else "")
        ctx = " | ".join(x for x in (self.DOC, a["part"], a["chapter"], a["sub"], label) if x)
        return self._emit(doc, a["paras"], "article", ctx, extra, id_base=f"art-{num}")

    # ------------------------------------------------------ annex + schedules
    def _chunk_tail(self, doc, text, base):
        chunks, section, kind, paras = [], None, None, []

        def close():
            if paras:
                slug = re.sub(r"[^a-z0-9]+", "-", section.lower()).strip("-")
                chunks.extend(self._emit(doc, paras, kind, f"{self.DOC} | {section}",
                                         {"section": section}, id_base=slug))

        for off, p in self._paras(text, base):
            if re.match(r"^_{3,}\s*ANNEX", p):
                close(); section, kind, paras = "Annex (Objectives Resolution)", "annex", []
                continue
            m = self.SCHEDULE.match(p)
            if m:
                close(); section, kind, paras = f"{m.group(1).title()} Schedule", "schedule", []
                paras.append((off, p)); continue
            if self.SEPARATOR.match(p):
                continue
            if section:
                paras.append((off, p))
        close()
        return chunks

    # ------------------------------------------------------ shared machinery
    @staticmethod
    def _paras(text, base=0):
        pos = 0
        for part in text.split("\n\n"):
            s = part.strip()
            if s:
                yield base + pos + (len(part) - len(part.lstrip())), s
            pos += len(part) + 2

    def _emit(self, doc, paras, kind, ctx, extra, id_base=None):
        paras = [(o, re.sub(r"\s*[_–—\-]{4,}\s*$", "", p)) for o, p in paras]   # trailing rule lines
        paras = [(o, p) for o, p in paras if p]
        if not paras:
            return []
        packed = self._pack(paras)
        out = []
        for k, (first_off, last_off, body) in enumerate(packed):
            p0, printed = doc.page_at(first_off)
            p1, _ = doc.page_at(last_off)
            notes, seen = [], set()
            for pg in range(p0, p1 + 1):
                for n in doc.footnotes_by_page.get(pg, []):
                    if n not in seen:
                        seen.add(n); notes.append(n)
            meta = {"doc": self.DOC, "kind": kind, "chunk_index": k, "chunk_count": len(packed),
                    "pdf_page": p0, **extra}
            if printed:
                meta["printed_page"] = printed
            if notes:
                meta["amendment_notes"] = " | ".join(notes)[:2000]
            header = ctx if len(packed) == 1 else f"{ctx} (part {k + 1}/{len(packed)})"
            cid = id_base or kind
            out.append(Chunk(id=cid if len(packed) == 1 else f"{cid}-p{k + 1}",
                             text=f"{header}\n\n{body}", metadata=meta))
        return out

    def _pack(self, paras):
        limit = self.max_chars
        pieces = []
        for off, p in paras:
            pieces += [(off, s) for s in (self._split_long(p, limit) if len(p) > limit else [p])]
        chunks, cur, n = [], [], 0
        for off, s in pieces:
            if cur and n + len(s) + 2 > limit:
                chunks.append(cur); cur, n = [], 0
            cur.append((off, s)); n += len(s) + 2
        if cur:
            chunks.append(cur)
        return [(c[0][0], c[-1][0], "\n\n".join(s for _, s in c)) for c in chunks]

    @staticmethod
    def _split_long(p, limit):
        """Split an oversized clause at sentence/semicolon boundaries, then at spaces as a last resort."""
        parts, cur = [], ""
        for sent in re.split(r"(?<=[.;:])\s+", p):
            while len(sent) > limit:
                cut = sent.rfind(" ", 0, limit)
                cut = cut if cut > 0 else limit
                if cur:
                    parts.append(cur); cur = ""
                parts.append(sent[:cut]); sent = sent[cut:].lstrip()
            if cur and len(cur) + len(sent) + 1 > limit:
                parts.append(cur); cur = sent
            else:
                cur = f"{cur} {sent}".strip()
        if cur:
            parts.append(cur)
        return parts

class ConstitutionIngestor:
    def __init__(self, extractor, cleaner, chunker, store=None):
        self.extractor, self.cleaner, self.chunker, self.store = extractor, cleaner, chunker, store

    def run(self, path, reset=False):
        pages = self.extractor.extract(path)
        doc = self.cleaner.clean(pages)

        log = getattr(self.cleaner, "hyphen_log", None)
        if log:
            print(f"{sum(log.values())} hyphen joins, {len(log)} distinct pairs:")
            for pair, n in sorted(log.items()):
                print(f"  {pair:28s} x{n}")

        chunks = self.chunker.chunk(doc)
        if self.store is not None:
            if reset:
                self.store.reset()
            self.store.upsert(chunks)
        return chunks
    

def main():
    ap = argparse.ArgumentParser(description="Ingest the Constitution of Pakistan into Chroma")
    ap.add_argument("pdf", nargs="?", default="constitution_of_Pakistan.pdf")
    ap.add_argument("--db-path", default="./chroma_db")
    ap.add_argument("--collection", default="pakistan_constitution")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--reset", action="store_true", help="drop the collection before ingesting")
    ap.add_argument("--dry-run", action="store_true", help="extract/clean/chunk only; skip embedding")
    args = ap.parse_args()

    store = None
    if not args.dry_run:
        from store import VectorStore   # imported lazily so --dry-run needs no chroma/torch
        store = VectorStore(path=args.db_path, collection=args.collection, model=args.model)

    chunks = ConstitutionIngestor(PDFExtractor(), TextCleaner(), ArticleChunker(), store).run(
        args.pdf, reset=args.reset)

    kinds = {}
    for c in chunks:
        kinds[c.metadata["kind"]] = kinds.get(c.metadata["kind"], 0) + 1
    print(f"{len(chunks)} chunks {kinds}")
    if store is not None:
        print(f"{store.collection.count()} stored in '{args.collection}' at {args.db_path}")


if __name__ == "__main__":
    main()