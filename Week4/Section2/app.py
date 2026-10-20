from flask import Flask, jsonify, request

from library_classes import (
    Book,
    DVD,
    ItemNotFoundError,
    ItemStateError,
    LibraryCollection,
)


app = Flask(__name__)

library = LibraryCollection()

library.add_item(Book("1984", "George Orwell", 328))
library.add_item(DVD("Interstellar", "Christopher Nolan", 169))
library.add_item(Book("The Hobbit", "J.R.R. Tolkien", 310))


@app.route("/api/items", methods=["GET"])
def list_items():
    available_param = request.args.get("available")
    item_type = request.args.get("type")

    available = None

    if available_param is not None:
        if available_param.lower() == "true":
            available = True
        elif available_param.lower() == "false":
            available = False
        else:
            return jsonify({"error": "Invalid available filter"}), 400

    try:
        items = library.list_items(
            available=available,
            item_type=item_type
        )
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    return jsonify([item.to_dict() for item in items]), 200


@app.route("/api/items/<int:item_id>", methods=["GET"])
def get_item(item_id):
    try:
        item = library.get_item(item_id)
    except ItemNotFoundError as error:
        return jsonify({"error": str(error)}), 404

    return jsonify(item.to_dict()), 200


@app.route("/api/items", methods=["POST"])
def create_item():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid input"}), 400

    try:
        item_type = data["type"].lower()
        title = data["title"]
        author = data["author"]

        if item_type == "book":
            page_count = data["page_count"]
            item = Book(title, author, page_count)

        elif item_type == "dvd":
            duration = data["duration"]
            item = DVD(title, author, duration)

        else:
            return jsonify({"error": "Invalid item type"}), 400

    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Invalid input"}), 400

    library.add_item(item)

    return jsonify(item.to_dict()), 201


@app.route("/api/items/<int:item_id>", methods=["PATCH"])
def patch_item(item_id):
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid input"}), 400

    try:
        item = library.update_item(item_id, **data)

    except ItemNotFoundError as error:
        return jsonify({"error": str(error)}), 404

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    return jsonify(item.to_dict()), 200


@app.route("/api/items/<int:item_id>", methods=["PUT"])
def replace_item(item_id):
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid input"}), 400

    try:
        item = library.get_item(item_id)

        required_fields = {"title", "author"}

        if isinstance(item, Book):
            required_fields.add("page_count")
        elif isinstance(item, DVD):
            required_fields.add("duration")

        if not required_fields.issubset(data.keys()):
            return jsonify({"error": "Missing required fields"}), 400

        editable_fields = {
            key: value
            for key, value in data.items()
            if key in {"title", "author", "page_count", "duration"}
        }

        item = library.update_item(item_id, **editable_fields)

    except ItemNotFoundError as error:
        return jsonify({"error": str(error)}), 404

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    return jsonify(item.to_dict()), 200


@app.route("/api/items/<int:item_id>", methods=["DELETE"])
def delete_item(item_id):
    try:
        library.remove_item(item_id)

    except ItemNotFoundError as error:
        return jsonify({"error": str(error)}), 404

    except ItemStateError as error:
        return jsonify({"error": str(error)}), 409

    return "", 204


@app.route("/api/items/<int:item_id>/checkout", methods=["POST"])
def checkout_item(item_id):
    try:
        item = library.get_item(item_id)
        message = item.checkout()

    except ItemNotFoundError as error:
        return jsonify({"error": str(error)}), 404

    except ItemStateError as error:
        return jsonify({"error": str(error)}), 409

    return jsonify({
        "message": message,
        "item": item.to_dict()
    }), 200


@app.route("/api/items/<int:item_id>/return", methods=["POST"])
def return_item(item_id):
    try:
        item = library.get_item(item_id)
        message = item.return_item()

    except ItemNotFoundError as error:
        return jsonify({"error": str(error)}), 404

    except ItemStateError as error:
        return jsonify({"error": str(error)}), 409

    return jsonify({
        "message": message,
        "item": item.to_dict()
    }), 200


if __name__ == "__main__":
    app.run(debug=True)