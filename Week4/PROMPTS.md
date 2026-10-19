# PROMPTS

## Prompt 1
Turn my Figma calculator design into a working web page using HTML, CSS, and JavaScript.

## Prompt 2
Create a Flask backend for the calculator and expose a POST /api/calculate endpoint.

## Prompt 3
Connect the frontend to the Flask API using fetch so that all calculations are performed on the backend.

## Corrections

### Correction 1
The calculator initially had no way to clear or remove entered values.
I added keyboard support:
- Backspace removes the last entered digit.
- Escape clears the calculator.

### Correction 2
I verified that calculation logic remains on the Flask backend and is not duplicated in JavaScript.

### Correction 3
I tested division by zero and confirmed that the backend error message is shown in the frontend.