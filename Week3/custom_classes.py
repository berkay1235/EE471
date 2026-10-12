class MediaItem:
    """Represent a generic media item in a library."""

    def __init__(self, title, author, is_available=True):
        """Initialize a media item with its title, author, and availability."""
        self.title = title
        self.author = author
        self.is_available = is_available

    def checkout(self):
        """Check out the item if it is currently available."""
        if self.is_available:
            self.is_available = False
            return "Successful checkout"

        return "Already out"

    def return_item(self):
        """Mark the item as available again."""
        self.is_available = True

    def __str__(self):
        """Return a readable description of the media item."""
        status = "Available" if self.is_available else "Checked Out"
        return f"{self.title} by {self.author} [{status}]"


class Book(MediaItem):
    """Represent a book with an additional page count."""

    def __init__(self, title, author, page_count, is_available=True):
        """Initialize a book with its page count."""
        super().__init__(title, author, is_available)
        self.page_count = page_count

    def __str__(self):
        """Return a readable description including the page count."""
        return f"{super().__str__()} - {self.page_count} pages"


class DVD(MediaItem):
    """Represent a DVD with an additional duration."""

    def __init__(self, title, author, duration, is_available=True):
        """Initialize a DVD with its duration."""
        super().__init__(title, author, is_available)
        self.duration = duration

    def checkout(self):
        """Check out the DVD and print a handling reminder if successful."""
        result = super().checkout()

        if result == "Successful checkout":
            print("Handle with care: Do not scratch the disc.")

        return result


class LibraryCollection:
    """Store and manage a collection of media items."""

    def __init__(self):
        """Initialize an empty library collection."""
        self.items = []

    def add_item(self, item):
        """Add a media item to the collection."""
        self.items.append(item)

    def list_available(self):
        """Return the titles of all currently available items."""
        return [
            item.title
            for item in self.items
            if item.is_available
        ]