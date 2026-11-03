class ItemNotFoundError(Exception):
    pass


class ItemStateError(Exception):
    pass


CHECKOUT_SUCCESS_MESSAGE = "Successful checkout"
RETURN_SUCCESS_MESSAGE = "Successful return"


class MediaItem:
    """Represent a generic media item in a library."""

    def __init__(self, title, author, is_available=True):
        self.id = None
        self.title = title
        self.author = author
        self.is_available = is_available

    def checkout(self):
        if not self.is_available:
            raise ItemStateError("Item is already checked out")

        self.is_available = False
        return CHECKOUT_SUCCESS_MESSAGE

    def return_item(self):
        if self.is_available:
            raise ItemStateError("Item is not checked out")

        self.is_available = True
        return RETURN_SUCCESS_MESSAGE

    def to_dict(self):
        return {
            "id": self.id,
            "type": "media",
            "title": self.title,
            "author": self.author,
            "is_available": self.is_available
        }

    def __str__(self):
        status = "Available" if self.is_available else "Checked Out"
        return f"{self.title} by {self.author} [{status}]"


class Book(MediaItem):
    """Represent a book with an additional page count."""

    def __init__(self, title, author, page_count, is_available=True):
        super().__init__(title, author, is_available)
        self.page_count = page_count

    def to_dict(self):
        data = super().to_dict()
        data["type"] = "book"
        data["page_count"] = self.page_count
        return data

    def __str__(self):
        return f"{super().__str__()} - {self.page_count} pages"


class DVD(MediaItem):
    """Represent a DVD with an additional duration."""

    def __init__(self, title, author, duration, is_available=True):
        super().__init__(title, author, is_available)
        self.duration = duration

    def to_dict(self):
        data = super().to_dict()
        data["type"] = "dvd"
        data["duration"] = self.duration
        return data

    def checkout(self):
        result = super().checkout()
        return f"{result}. Handle with care: Do not scratch the disc."


class LibraryCollection:
    """Store and manage a collection of media items."""

    def __init__(self):
        self.items = []
        self._next_id = 1

    def add_item(self, item):
        item.id = self._next_id
        self._next_id += 1
        self.items.append(item)
        return item.id

    def get_item(self, item_id):
        for item in self.items:
            if item.id == item_id:
                return item

        raise ItemNotFoundError(f"Item with id {item_id} not found")

    def list_items(self, available=None, item_type=None):
        result = self.items

        if available is not None:
            result = [
                item
                for item in result
                if item.is_available == available
            ]

        if item_type is not None:
            item_type = item_type.lower()

            if item_type not in ("book", "dvd", "media"):
                raise ValueError("Invalid item type")

            result = [
                item
                for item in result
                if item.to_dict()["type"] == item_type
            ]

        return result

    def update_item(self, item_id, **fields):
        item = self.get_item(item_id)

        forbidden_fields = {"id", "type", "is_available"}

        for field in fields:
            if field in forbidden_fields:
                raise ValueError(f"Field '{field}' cannot be updated")

            if field not in {"title", "author", "page_count", "duration"}:
                raise ValueError(f"Invalid field '{field}'")

        if "page_count" in fields and not isinstance(item, Book):
            raise ValueError("page_count can only be updated for books")

        if "duration" in fields and not isinstance(item, DVD):
            raise ValueError("duration can only be updated for DVDs")

        for field, value in fields.items():
            setattr(item, field, value)

        return item

    def remove_item(self, item_id):
        item = self.get_item(item_id)

        if not item.is_available:
            raise ItemStateError("Cannot remove a checked-out item")

        self.items.remove(item)

    def list_available(self):
        return [
            item.title
            for item in self.items
            if item.is_available
        ]