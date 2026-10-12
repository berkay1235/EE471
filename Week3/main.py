from custom_classes import MediaItem


book = MediaItem("1984", "George Orwell")
movie = MediaItem("Interstellar", "Christopher Nolan")

print(book)
print(movie)

print(book.checkout())

print(book)
print(movie)