const itemList = document.getElementById("itemList");
const emptyState = document.getElementById("emptyState");
const errorBox = document.getElementById("errorBox");

const availableOnly = document.getElementById("availableOnly");
const typeFilter = document.getElementById("typeFilter");

const itemForm = document.getElementById("itemForm");
const formTitle = document.getElementById("formTitle");
const editId = document.getElementById("editId");

const titleInput = document.getElementById("title");
const authorInput = document.getElementById("author");
const typeInput = document.getElementById("type");
const pageCountInput = document.getElementById("pageCount");
const durationInput = document.getElementById("duration");

const pageCountGroup = document.getElementById("pageCountGroup");
const durationGroup = document.getElementById("durationGroup");

const cancelEditButton = document.getElementById("cancelEdit");


function showError(message) {
    errorBox.textContent = message;
}


function clearError() {
    errorBox.textContent = "";
}


function resetForm() {
    itemForm.reset();
    editId.value = "";
    formTitle.textContent = "Add Item";

    typeInput.value = "book";
    pageCountGroup.classList.remove("hidden");
    durationGroup.classList.add("hidden");

    clearError();
}


function updateTypeFields() {
    if (typeInput.value === "book") {
        pageCountGroup.classList.remove("hidden");
        durationGroup.classList.add("hidden");
        durationInput.value = "";
    } else {
        pageCountGroup.classList.add("hidden");
        durationGroup.classList.remove("hidden");
        pageCountInput.value = "";
    }
}


async function loadItems() {
    clearError();

    let url = "/api/items";
    const params = new URLSearchParams();

    if (availableOnly.checked) {
        params.append("available", "true");
    }

    if (typeFilter.value) {
        params.append("type", typeFilter.value);
    }

    if (params.toString()) {
        url += `?${params.toString()}`;
    }

    try {
        const response = await fetch(url);
        const data = await response.json();

        if (!response.ok) {
            showError(data.error);
            return;
        }

        renderItems(data);
    } catch (error) {
        showError("Could not connect to the server.");
    }
}


function renderItems(items) {
    itemList.innerHTML = "";

    if (items.length === 0) {
        emptyState.style.display = "block";
        return;
    }

    emptyState.style.display = "none";

    items.forEach((item) => {
        const card = document.createElement("div");
        card.className = "item-card";

        const detail =
            item.type === "book"
                ? `${item.page_count} pages`
                : `${item.duration} minutes`;

        const status = item.is_available
            ? "Available"
            : "Checked Out";

        const stateButton = item.is_available
            ? `<button class="checkout-btn" onclick="checkoutItem(${item.id})">Checkout</button>`
            : `<button class="return-btn" onclick="returnItem(${item.id})">Return</button>`;

        card.innerHTML = `
            <h3>${item.title}</h3>
            <p class="item-meta">${item.author}</p>
            <p class="item-meta">${item.type.toUpperCase()} • ${detail}</p>
            <span class="badge">${status}</span>

            <div class="item-actions">
                ${stateButton}
                <button class="edit-btn" onclick="editItem(${item.id})">Edit</button>
                <button class="delete-btn" onclick="deleteItem(${item.id})">Delete</button>
            </div>
        `;

        itemList.appendChild(card);
    });
}


async function checkoutItem(id) {
    clearError();

    const response = await fetch(`/api/items/${id}/checkout`, {
        method: "POST"
    });

    const data = await response.json();

    if (!response.ok) {
        showError(data.error);
        return;
    }

    loadItems();
}


async function returnItem(id) {
    clearError();

    const response = await fetch(`/api/items/${id}/return`, {
        method: "POST"
    });

    const data = await response.json();

    if (!response.ok) {
        showError(data.error);
        return;
    }

    loadItems();
}


async function deleteItem(id) {
    clearError();

    const response = await fetch(`/api/items/${id}`, {
        method: "DELETE"
    });

    if (!response.ok) {
        const data = await response.json();
        showError(data.error);
        return;
    }

    loadItems();
}


async function editItem(id) {
    clearError();

    const response = await fetch(`/api/items/${id}`);
    const item = await response.json();

    if (!response.ok) {
        showError(item.error);
        return;
    }

    editId.value = item.id;
    titleInput.value = item.title;
    authorInput.value = item.author;
    typeInput.value = item.type;

    formTitle.textContent = "Edit Item";

    updateTypeFields();

    if (item.type === "book") {
        pageCountInput.value = item.page_count;
    } else {
        durationInput.value = item.duration;
    }
}


itemForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearError();

    const itemType = typeInput.value;

    const payload = {
        title: titleInput.value,
        author: authorInput.value,
        type: itemType
    };

    if (itemType === "book") {
        payload.page_count = Number(pageCountInput.value);
    } else {
        payload.duration = Number(durationInput.value);
    }

    let url = "/api/items";
    let method = "POST";

    if (editId.value) {
        url = `/api/items/${editId.value}`;
        method = "PUT";

        delete payload.type;
    }

    const response = await fetch(url, {
        method: method,
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
    });

    const data = await response.json();

    if (!response.ok) {
        showError(data.error);
        return;
    }

    resetForm();
    loadItems();
});


availableOnly.addEventListener("change", loadItems);
typeFilter.addEventListener("change", loadItems);
typeInput.addEventListener("change", updateTypeFields);

cancelEditButton.addEventListener("click", resetForm);


updateTypeFields();
loadItems();