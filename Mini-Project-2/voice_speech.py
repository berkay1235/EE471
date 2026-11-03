"""Build the sentence the librarian says after a command has run.

Input: the parsed Command plus what the REST API answered (status + JSON body).
Output: one plain-text sentence for Azure text-to-speech.
"""

MAX_SPOKEN_ITEMS = 5   # a long list would take forever to read out loud

NEEDS_ID = {"get", "replace", "update", "delete", "checkout", "return"}

HELP = (
    "Sorry, I didn't understand that. You can say, for example: "
    "check out item 3, or, list available books."
)

CREATE_HINT = (
    "Please say the type, title, author, and page count or duration. For example: "
    "create item, type book, title Dune, author Frank Herbert, pages 412."
)

ASK_FOR_ID = {
    "get": "get item 3",
    "replace": "replace item 3, title Dune, author Frank Herbert, pages 412",
    "update": "update item 3, pages 250",
    "delete": "delete item 3",
    "checkout": "check out item 3",
    "return": "return item 3",
}


def type_label(item_type):
    return "DVD" if item_type == "dvd" else "book"


def error_text(body):
    """'Item is already checked out' -> 'item is already checked out'."""
    message = (body or {}).get("error", "something went wrong")
    message = message.replace("_", " ")   # "page_count" would be read as "page underscore count"
    if len(message) > 1 and message[0].isupper() and not message[1].isupper():
        message = message[0].lower() + message[1:]
    return message


def describe_filter(filters, count):
    """('available', 'books') -> 'available books'; singular when count == 1."""
    singular, plural = {"book": ("book", "books"), "dvd": ("DVD", "DVDs")}.get(
        filters.get("type"), ("item", "items")
    )
    state = {"true": "available ", "false": "checked out "}.get(filters.get("available"), "")
    return f"{state}{singular if count == 1 else plural}"


def describe_item(item):
    """'the book 'Dune' by Frank Herbert, 412 pages'"""
    if item["type"] == "dvd":
        return (f"the DVD '{item['title']}' directed by {item['author']}, "
                f"{item['duration']} minutes")
    return f"the book '{item['title']}' by {item['author']}, {item['page_count']} pages"


def speech_for(command, status, body, item_before=None):
    intent = command.intent
    item_id = command.item_id

    if intent == "unknown":
        return HELP

    if intent in NEEDS_ID and item_id is None:
        return f"Sorry, which item? For example, say: {ASK_FOR_ID[intent]}."

    if status == 404:
        return f"Sorry, item {item_id} was not found."

    # ---- create (POST /api/items) -------------------------------------
    if intent == "create":
        if status == 201:
            return (f"Successfully created {type_label(body['type'])} "
                    f"'{body['title']}' with ID {body['id']}.")
        return f"Sorry, I couldn't create that item. {CREATE_HINT}"

    # ---- list (GET /api/items) ----------------------------------------
    if intent == "list":
        if status != 200:
            return f"Sorry, {error_text(body)}."
        what = describe_filter(command.filters, len(body))
        if not body:
            return f"There are no {what}."

        spoken = [f"{item['id']}, {item['title']} by {item['author']}"
                  for item in body[:MAX_SPOKEN_ITEMS]]
        extra = len(body) - MAX_SPOKEN_ITEMS
        verb = "is" if len(body) == 1 else "are"
        sentence = f"There {verb} {len(body)} {what}: " + "; ".join(spoken) + "."
        if extra > 0:
            sentence += f" And {extra} more."
        return sentence

    # ---- get (GET /api/items/<id>) ------------------------------------
    if intent == "get":
        if status != 200:
            return f"Sorry, {error_text(body)}."
        state = "available" if body["is_available"] else "checked out"
        return f"Item {item_id} is {describe_item(body)}. It is {state}."

    # ---- replace / update (PUT / PATCH) -------------------------------
    if intent in ("replace", "update"):
        if status != 200:
            return f"Sorry, {error_text(body)}."
        verb = "replaced" if intent == "replace" else "updated"
        return f"Successfully {verb} item {item_id}, '{body['title']}'."

    # ---- delete (DELETE /api/items/<id>) ------------------------------
    if intent == "delete":
        if status == 204:
            title = item_before["title"] if item_before else None
            if title:
                return f"Item {item_id}, '{title}', has been deleted."
            return f"Item {item_id} has been deleted."
        if status == 409:
            return f"Sorry, item {item_id} is checked out and cannot be deleted."
        return f"Sorry, {error_text(body)}."

    # ---- checkout (POST /api/items/<id>/checkout) ---------------------
    if intent == "checkout":
        if status == 200:
            item = body["item"]
            sentence = f"Item {item_id}, '{item['title']}', has been checked out."
            if item["type"] == "dvd":
                sentence += " Please handle the disc with care."
            return sentence
        if status == 409:
            return f"Sorry, item {item_id} is already checked out."
        return f"Sorry, {error_text(body)}."

    # ---- return (POST /api/items/<id>/return) -------------------------
    if intent == "return":
        if status == 200:
            return f"Item {item_id}, '{body['item']['title']}', has been returned."
        if status == 409:
            return f"Sorry, item {item_id} is not checked out."
        return f"Sorry, {error_text(body)}."

    return HELP