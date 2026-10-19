from flask import Flask, jsonify, render_template, request

from custom_classes import Calculator


app = Flask(__name__)
calculator = Calculator()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/calculate", methods=["POST"])
def calculate():
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid input"}), 400

    try:
        x = float(data["x"])
        y = float(data["y"])
        operation = data["operation"]
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Invalid input"}), 400

    try:
        if operation == "+":
            result = calculator.add(x, y)
        elif operation == "-":
            result = calculator.subtract(x, y)
        elif operation == "*":
            result = calculator.multiply(x, y)
        elif operation == "/":
            result = calculator.divide(x, y)
        else:
            return jsonify({"error": "Invalid operation"}), 400
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    return jsonify({"result": result}), 200


if __name__ == "__main__":
    app.run(debug=True)