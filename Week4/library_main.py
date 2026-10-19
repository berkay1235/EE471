from library_classes import (
    Book,
    DVD,
    MediaItem,
    LibraryCollection,
    ItemNotFoundError,
    ItemStateError,
)


library = LibraryCollection()

book = Book("1984", "George Orwell", 328)
dvd = DVD("Interstellar", "Christopher Nolan", 169)
media = MediaItem("Inception", "Christopher Nolan")

book_id = library.add_item(book)
dvd_id = library.add_item(dvd)
media_id = library.add_item(media)

print("Added item IDs:")
print(book_id, dvd_id, media_id)

print("\nAll items:")
for item in library.list_items():
    print(item.to_dict())

print("\nGet item:")
print(library.get_item(book_id).to_dict())

print("\nUpdate item:")
updated_book = library.update_item(book_id, title="Nineteen Eighty-Four")
print(updated_book.to_dict())

print("\nCheckout DVD:")
print(dvd.checkout())

print("\nAvailable items:")
for item in library.list_items(available=True):
    print(item.to_dict())

print("\nDVD items:")
for item in library.list_items(item_type="dvd"):
    print(item.to_dict())

print("\nReturn DVD:")
print(dvd.return_item())

print("\nRemove media item:")
library.remove_item(media_id)

for item in library.list_items():
    print(item.to_dict())


print("\nTesting ItemNotFoundError:")
try:
    library.get_item(999)
except ItemNotFoundError as error:
    print(error)


print("\nTesting ItemStateError for checkout:")
try:
    book.checkout()
    book.checkout()
except ItemStateError as error:
    print(error)


print("\nTesting ItemStateError for return:")
try:
    dvd.return_item()
except ItemStateError as error:
    print(error)


print("\nTesting ValueError:")
try:
    library.update_item(book_id, id=100)
except ValueError as error:
    print(error)