"""Turn a spoken sentence (the text Azure Speech heard) into a Command."""

import re
from dataclasses import dataclass, field


@dataclass
class Command:
    intent: str                                  # create | list | get | replace | update
                                                 # delete | checkout | return | unknown
    item_id: int | None = None
    fields: dict = field(default_factory=dict)   # body for create / replace / update
    filters: dict = field(default_factory=dict)  # query string for list
    transcript: str = ""


# ---------- Numbers: "three" -> 3, "twenty two" -> 22 ----------
ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
# Speech recognition sometimes writes these instead of the digit ("item for" = item 4)
HOMOPHONES = {"to": 2, "too": 2, "for": 4, "won": 1, "ate": 8, "tree": 3}


def words_to_int(tokens):
    """['two', 'hundred', 'twenty'] -> 220. Returns None if it is not a number."""
    if not tokens:
        return None
    current = 0
    for token in tokens:
        if token in ONES:
            current += ONES[token]
        elif token in TENS:
            current += TENS[token]
        elif token == "hundred":
            current = max(current, 1) * 100
        else:
            return None
    return current


def extract_item_id(text):
    """Find the item number in 'check out item 3' / 'item three' / 'number 3'."""
    lowered = text.lower()

    # Digits: "item 3", "item number 3", "id 3", "#3"
    match = re.search(
        r"(?:\bitem\b|\bid\b|\bnumber\b|#)\s*(?:number|no\.?|id|#)?\s*[:#]?\s*(\d+)", lowered
    )
    if match:
        return int(match.group(1))

    # Spoken words: "item three", "item twenty two"
    match = re.search(r"\b(?:item|id|number)\b[\s:#,]*([a-z\- ]+)", lowered)
    if match:
        tokens = re.findall(r"[a-z]+", match.group(1))[:4]
        for length in range(len(tokens), 0, -1):
            value = words_to_int(tokens[:length])
            if value is not None:
                return value
        if tokens and tokens[0] in HOMOPHONES:
            return HOMOPHONES[tokens[0]]

    # A bare number: "check out 3"
    match = re.search(r"\b(\d+)\b", lowered)
    return int(match.group(1)) if match else None


# ---------- Field keywords: title, author/director, type, pages, duration ----------
TEXT_FIELDS = {
    "title": "title", "author": "author", "director": "author",
    "writer": "author", "type": "type",
}
NUMBER_FIELDS = {
    "page count": "page_count", "pages": "page_count", "page": "page_count",
    "duration": "duration", "minutes": "duration", "runtime": "duration",
    "length": "duration",
}

# Number keywords only count when a number follows ("pages 220"), so a title
# like "The Page Turner" is not cut in half.
KEYWORD_RE = re.compile(
    r"\b(title|author|director|writer|type)\b"
    r"|\b(page count|pages|page|duration|minutes|runtime|length)\b"
    r"(?=\s*(?:is|to|of|equals|:|=)?\s*\d)",
    re.IGNORECASE,
)


def normalize(text):
    """Small clean-ups before parsing."""
    text = " ".join(text.split())
    # "D V D", "D.V.D." -> "DVD"
    text = re.sub(r"\bd\.?\s*v\.?\s*d\.?s?\b", "DVD", text, flags=re.IGNORECASE)
    # "220 pages" / "148 minutes" -> "pages 220" / "duration 148"
    # (skipped when another number follows, e.g. title "1984 pages 328")
    text = re.sub(
        r"\b(\d+)\s*pages?\b(?!\s*(?:is|to|of|equals|:|=)?\s*\d)", r"pages \1", text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(\d+)\s*(?:minutes?|mins?)\b(?!\s*(?:is|to|of|equals|:|=)?\s*\d)", r"duration \1",
        text, flags=re.IGNORECASE,
    )
    # "change the title of item 3 to Emma" -> "update item 3 title Emma"
    text = re.sub(
        r"^\s*(?:change|update|edit|set|modify)\s+(?:the\s+)?"
        r"(title|author|director|pages?|page count|duration|length|runtime)\s+of\s+item\s+"
        r"(\w+)\s+(?:to|as)\s+",
        r"update item \2 \1 ",
        text, flags=re.IGNORECASE,
    )
    return text


def clean_text_value(raw, strip_connector):
    value = raw.strip()
    # "title is Dune", "title to Dune" ("to" is only a connector when editing)
    connectors = r"(?:is|to|as|equals)" if strip_connector else r"(?:is)"
    value = re.sub(rf"^(?:{connectors}\b|[:=])\s*", "", value, flags=re.IGNORECASE)
    value = value.strip(" ,;:!?\"'“”‘’")
    value = re.sub(r"(?:\s+(?:and|with|then))+$", "", value, flags=re.IGNORECASE)
    value = value.strip(" ,;:!?\"'“”‘’")
    # Drop the sentence-ending period, but keep initials such as "J.D."
    if value.endswith(".") and not re.search(r"(?:^|[\s.])[A-Za-z]\.$", value):
        value = value[:-1]
    return value.strip()


def parse_number(raw):
    match = re.search(r"\d+", raw)
    if match:
        return int(match.group())
    return words_to_int(re.findall(r"[a-z]+", raw.lower()))


def parse_type(raw):
    lowered = raw.lower()
    if "dvd" in lowered or re.search(r"\b(movie|film|disc|disk)\b", lowered):
        return "dvd"
    if "book" in lowered:
        return "book"
    return None


def extract_fields(text, strip_connector):
    """Split 'type book, title X, author Y, pages 220' into a dict."""
    fields = {}
    matches = list(KEYWORD_RE.finditer(text))

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        raw = text[match.end():end]

        if match.group(1):                       # text keyword
            name = TEXT_FIELDS[match.group(1).lower()]
            if name == "type":
                value = parse_type(raw)
            else:
                value = clean_text_value(raw, strip_connector)
        else:                                    # number keyword
            name = NUMBER_FIELDS[match.group(2).lower()]
            value = parse_number(raw)

        if value not in (None, ""):
            fields[name] = value

    return fields


# ---------- Intent ----------
def detect_intent(head, has_id):
    padded = " " + re.sub(r"[^a-z0-9# ]+", " ", head.lower()) + " "

    def has(pattern):
        return re.search(pattern, padded) is not None

    if has(r"\b(check\s?out|checkout|borrow)\b"):
        return "checkout"
    if has(r"\b(return|check\s?in|checkin)\b"):
        return "return"
    if has(r"\b(delete|remove|erase|discard)\b"):
        return "delete"
    if has(r"\breplace\b"):
        return "replace"
    if has(r"\b(update|change|edit|modify|rename|set|patch)\b"):
        return "update"
    if has(r"\b(create|add|new|make|insert|register)\b"):
        return "create"
    if has_id and has(r"\b(get|show|tell|describe|read|details|info|open|find|display|look)\b"):
        return "get"
    if has(r"\b(list|show|display)\b"):
        return "list"
    # "What books are available?" needs a noun too, otherwise any
    # sentence starting with "what" would count as a list command
    if has(r"\b(what|which|all|available|everything)\b") and has(
        r"\b(items?|books?|dvds?|movies?|films?|library|everything)\b"
    ):
        return "list"
    if has(r"\b(items|books|dvds)\b"):
        return "list"
    return "unknown"


def extract_filters(text):
    """'list available books' -> {'available': 'true', 'type': 'book'}."""
    lowered = text.lower()
    filters = {}

    if re.search(r"\b(checked out|not available|unavailable|borrowed)\b", lowered):
        filters["available"] = "false"
    elif re.search(r"\b(available|on the shelf|in stock)\b", lowered):
        filters["available"] = "true"

    if re.search(r"\bdvds?\b|\bmovies?\b|\bfilms?\b", lowered):
        filters["type"] = "dvd"
    elif re.search(r"\bbooks?\b", lowered):
        filters["type"] = "book"

    return filters


def parse_command(transcript):
    text = normalize(transcript or "")

    # Everything before the first field keyword is the "command part"
    first_keyword = KEYWORD_RE.search(text)
    head = text[:first_keyword.start()] if first_keyword else text

    item_id = extract_item_id(head)
    intent = detect_intent(head, item_id is not None)
    command = Command(intent=intent, item_id=item_id, transcript=transcript)

    if intent in ("create", "replace", "update"):
        command.fields = extract_fields(text, strip_connector=intent != "create")

    if intent == "create":
        # No explicit "type": guess it from the sentence or from pages / duration
        if "type" not in command.fields:
            guessed = parse_type(head)
            if guessed is None:
                if "page_count" in command.fields:
                    guessed = "book"
                elif "duration" in command.fields:
                    guessed = "dvd"
            if guessed:
                command.fields["type"] = guessed
        command.item_id = None            # the backend assigns the id

    elif intent == "list":
        command.item_id = None
        command.filters = extract_filters(text)

    return command