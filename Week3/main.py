from custom_classes import Book, DVD, MediaItem, LibraryCollection


library = LibraryCollection()

book = Book("1984", "George Orwell", 328)
dvd = DVD("Interstellar", "Christopher Nolan", 169)
movie = MediaItem("Inception", "Christopher Nolan")

library.add_item(book)
library.add_item(dvd)
library.add_item(movie)

book.checkout()
dvd.checkout()

print("Available items:")
print(library.list_available())