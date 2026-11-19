"""Splits a two-part question so each part is answered on its own.

The pipeline answers a question with ONE query that returns ONE table.
"Show expenses by location and category. Also identify which location had
the highest total." asks for two tables of different shapes. Given both
parts at once, the model sometimes folds the second into the first (for
example by adding ``LIMIT 1`` to the breakdown), which returns a result
that answers neither.

So the split is made in code, before any model sees the question: the
Planner and the Programmer work on the first part only, and the second
part is handed back to the user as a follow-up question.

The rule is deliberately narrow. A split happens only when a later sentence

* starts with an additive cue ("Also", "Additionally", "In addition",
  "Furthermore", "Plus"), and
* asks something of its own: it has a question word ("which", "what",
  "who", "how many", "how much") or a superlative ("highest", "lowest",
  "most", "top", "overall", ...), and
* does not start with a word that merely adjusts the first result
  ("include", "sort", "rank", "exclude", "limit", ...).

"Also include the city" or "Also sort them by revenue" are left alone, so
the single table can simply gain a column or a sort order.

Inside a sentence ("..., and also identify which location was highest")
the rule is stricter: the second part must contain a question word. "and
also" usually joins two columns of one table ("revenue and also the
highest price"), and those stay together.
"""

import re
from typing import List, Tuple

_CUE = r"(?:and\s+)?(?:also|additionally|in\s+addition|furthermore|plus)\b"

# A sentence end (or a line break) followed by a cue word.
_SENTENCE_BOUNDARY = rf"(?P<sentence>[.?!;]\s+|\s*\n\s*)(?={_CUE})"
# The same cues in the middle of a sentence: ", and also ...", ", also ...".
_INLINE_BOUNDARY = r"(?P<inline>,?\s+(?=and\s+also\b)|,\s+(?=(?:also|plus)\b))"
_BOUNDARY = re.compile(rf"{_SENTENCE_BOUNDARY}|{_INLINE_BOUNDARY}", re.IGNORECASE)
_LEADING_CUE = re.compile(rf"^{_CUE}[\s,:\-–—]*", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.?!])\s+")
_WORD = re.compile(r"[A-Za-z0-9']+")

_QUESTION_WORD = re.compile(
    r"\b(?:which|what|who|whom|whose|how\s+many|how\s+much)\b", re.IGNORECASE
)
_SUPERLATIVE = re.compile(
    r"\b(?:highest|lowest|most|least|top|bottom|best|worst|largest|smallest"
    r"|biggest|maximum|minimum|overall)\b",
    re.IGNORECASE,
)

# First words of a sentence that adjusts the first result instead of asking
# for a second one.
_ADJUSTS_FIRST_RESULT = frozenset({
    "include", "add", "exclude", "restrict", "limit", "filter", "use",
    "assume", "only", "ignore", "consider", "sort", "order", "rank", "group",
    "break", "split", "round", "express", "format", "make", "ensure", "keep",
    "remove", "drop", "label",
})

# First words of a sentence that sets the scene ("Use 2024 data.") and is
# not a question in itself.
_SETS_THE_SCENE = frozenset({
    "use", "assume", "consider", "define", "note", "given", "suppose", "let",
    "treat", "ignore", "exclude", "only",
})

_MIN_MAIN_WORDS = 3
_MIN_FOLLOW_UP_WORDS = 2


def _words(text: str) -> List[str]:
    return _WORD.findall(text.lower())


def _asks_a_question(main: str) -> bool:
    """Checks that the first part holds a request, not just scene-setting.

    Args:
        main: The text before the split.

    Returns:
        True if, after leaving out sentences such as "Use 2024 data.", at
        least a few words remain.
    """
    remaining = 0
    for sentence in _SENTENCE_END.split(main):
        words = _words(sentence)
        if words and words[0] not in _SETS_THE_SCENE:
            remaining += len(words)
    return remaining >= _MIN_MAIN_WORDS


def split_question(question: str) -> Tuple[str, str]:
    """Separates a question into its main part and a follow-up part.

    Args:
        question: The question as the user typed it.

    Returns:
        A tuple of (main part, follow-up part). The follow-up has its cue
        word and closing punctuation removed. It is an empty string, and
        the main part is the whole question, when there is nothing to split.
    """
    text = (question or "").strip()
    for boundary in _BOUNDARY.finditer(text):
        inline = boundary.group("inline") is not None
        main = text[:boundary.end()].strip().rstrip(";,").strip()
        follow_up = _LEADING_CUE.sub("", text[boundary.end():].strip()).strip()
        follow_up = follow_up.rstrip(".;!? \t\n")
        follow_up_words = _words(follow_up)

        if len(follow_up_words) < _MIN_FOLLOW_UP_WORDS:
            continue
        if follow_up_words[0] in _ADJUSTS_FIRST_RESULT:
            continue
        asks_something = _QUESTION_WORD.search(follow_up) or (
            not inline and _SUPERLATIVE.search(follow_up)
        )
        if not asks_something:
            continue
        if not _asks_a_question(main):
            continue
        return main, follow_up
    return text, ""
