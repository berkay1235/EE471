const display = document.getElementById("display");
const errorBox = document.getElementById("error");
const numberButtons = document.querySelectorAll(".number");
const operatorButtons = document.querySelectorAll(".operator");
const equalsButton = document.getElementById("equals");

let firstValue = "";
let secondValue = "";
let operation = null;
let enteringSecondValue = false;

function updateDisplay(value) {
    display.textContent = value;
}

function clearError() {
    errorBox.textContent = "";
}

numberButtons.forEach((button) => {
    button.addEventListener("click", () => {
        clearError();

        const value = button.textContent;

        if (!enteringSecondValue) {
            firstValue += value;
            updateDisplay(firstValue);
        } else {
            secondValue += value;
            updateDisplay(secondValue);
        }
    });
});

operatorButtons.forEach((button) => {
    button.addEventListener("click", () => {
        clearError();

        if (firstValue === "") {
            errorBox.textContent = "Enter the first number.";
            return;
        }

        operation = button.textContent;
        enteringSecondValue = true;
    });
});

equalsButton.addEventListener("click", async () => {
    clearError();

    if (firstValue === "" || secondValue === "" || operation === null) {
        errorBox.textContent = "Invalid input.";
        return;
    }

    try {
        const response = await fetch("/api/calculate", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                x: firstValue,
                y: secondValue,
                operation: operation
            })
        });

        const data = await response.json();

        if (!response.ok) {
            errorBox.textContent = data.error;
            return;
        }

        updateDisplay(data.result);

        firstValue = String(data.result);
        secondValue = "";
        operation = null;
        enteringSecondValue = false;
    } catch (error) {
        errorBox.textContent = "Server error.";
    }
});
document.addEventListener("keydown", (event) => {
    if (event.key === "Backspace") {
        clearError();

        if (enteringSecondValue) {
            secondValue = secondValue.slice(0, -1);
            updateDisplay(secondValue || "0");
        } else {
            firstValue = firstValue.slice(0, -1);
            updateDisplay(firstValue || "0");
        }
    }

    if (event.key === "Escape") {
        firstValue = "";
        secondValue = "";
        operation = null;
        enteringSecondValue = false;

        clearError();
        updateDisplay("0");
    }
});