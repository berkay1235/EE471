# PROMPTS

## Prompt 1
Create a single-page library web application using HTML, CSS, and JavaScript.

## Prompt 2
Connect the frontend to the existing Flask REST API using fetch.

## Prompt 3
Implement item listing, filtering, add, edit, delete, checkout, and return operations through the API.

## Corrections

### Correction 1
The frontend must not duplicate library business rules in JavaScript.
All item state logic remains in the Flask backend and library classes.

### Correction 2
Backend API error messages must be displayed in the frontend.
For example, deleting a checked-out item shows the 409 error message.

### Correction 3
The form must switch between page count for books and duration for DVDs.