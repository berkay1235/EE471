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


// Titles can now come from the microphone, so never insert them as raw HTML
function escapeHtml(text) {
    return String(text)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
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
            <h3>${escapeHtml(item.title)} <span class="id-badge">#${item.id}</span></h3>
            <p class="item-meta">${escapeHtml(item.author)}</p>
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


// =====================================================================
// Voice (Mini Project 2)
//   microphone --Azure STT--> transcript --POST /api/voice--> sentence
//   sentence --POST /api/speak (Azure TTS)--> audio played in the browser
// =====================================================================
const micButton = document.getElementById("mic");
const voiceStatus = document.getElementById("voiceStatus");
const heardBox = document.getElementById("heard");
const saidBox = document.getElementById("said");
const traceBox = document.getElementById("trace");
const commandForm = document.getElementById("commandForm");
const commandInput = document.getElementById("commandInput");

const IDLE_MESSAGE = "Click the microphone and say a command, e.g. “check out item 1”.";
let busy = false;   // true while listening, running a command or speaking


function setStatus(message, failed = false) {
    voiceStatus.textContent = message;
    voiceStatus.classList.toggle("failed", failed);
}


function setBusy(value, listening = false) {
    busy = value;
    micButton.disabled = value && !listening;
    micButton.classList.toggle("listening", listening);
}


// ---------- 1) Speaking (Text-to-Speech through Flask /api/speak) ----------
// Web Audio is used so Safari also plays audio that arrives after a fetch.
let audioCtx = null;

function unlockAudio() {
    // Browsers only allow sound after a user action, so the AudioContext
    // is created / resumed inside the click handler.
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === "suspended") {
        audioCtx.resume();
    }
}


async function speak(text) {
    const response = await fetch("/api/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
    });

    if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || "Text-to-speech failed");
    }

    const buffer = await audioCtx.decodeAudioData(await response.arrayBuffer());

    await new Promise((resolve) => {
        const source = audioCtx.createBufferSource();
        source.buffer = buffer;
        source.connect(audioCtx.destination);
        source.onended = resolve;
        source.start();
    });
}


// ---------- 2) Listening (real-time Speech-to-Text, one sentence) ----------
async function getSpeechToken() {
    const response = await fetch("/api/speech-token");
    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.error || "Could not get a speech token");
    }

    return data;   // { token, region }
}


async function listenOnce() {
    if (!window.SpeechSDK) {
        throw new Error("Azure Speech SDK could not be loaded (check your internet connection).");
    }

    const { token, region } = await getSpeechToken();

    const speechConfig = SpeechSDK.SpeechConfig.fromAuthorizationToken(token, region);
    speechConfig.speechRecognitionLanguage = "en-US";

    // A command has natural pauses ("... type book, title ..."), so wait a bit
    // longer than the default before deciding the sentence is finished.
    if (SpeechSDK.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs !== undefined) {
        speechConfig.setProperty(SpeechSDK.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "1500");
    }
    speechConfig.setProperty(SpeechSDK.PropertyId.Speech_SegmentationSilenceTimeoutMs, "1500");

    const audioConfig = SpeechSDK.AudioConfig.fromDefaultMicrophoneInput();
    const recognizer = new SpeechSDK.SpeechRecognizer(speechConfig, audioConfig);

    // Phrase list: tells Azure which words to expect, so "check out" is not heard as "chuck out"
    const phraseList = SpeechSDK.PhraseListGrammar.fromRecognizer(recognizer);
    [
        "check out", "return", "delete", "replace", "update", "create item", "list items",
        "item", "DVD", "book", "title", "author", "director", "pages", "duration", "available"
    ].forEach((phrase) => phraseList.addPhrase(phrase));

    return new Promise((resolve, reject) => {
        recognizer.recognizeOnceAsync(
            (result) => {
                recognizer.close();

                if (result.reason === SpeechSDK.ResultReason.RecognizedSpeech) {
                    resolve(result.text);
                } else if (result.reason === SpeechSDK.ResultReason.NoMatch) {
                    resolve("");
                } else {
                    const details = SpeechSDK.CancellationDetails.fromResult(result);
                    reject(new Error(details.errorDetails || "Speech was not recognized."));
                }
            },
            (error) => {
                recognizer.close();
                reject(new Error(String(error)));
            }
        );
    });
}


// ---------- 3) Run the command and say the result ----------
function showTrace(trace) {
    traceBox.innerHTML = trace
        .map((call) => {
            const ok = call.status < 400;
            return `<div class="${ok ? "" : "bad"}">${escapeHtml(call.method)} ${escapeHtml(call.path)} → ${call.status}</div>`;
        })
        .join("");
}


async function runTranscript(transcript) {
    heardBox.textContent = `Heard: “${transcript}”`;
    saidBox.textContent = "";
    traceBox.textContent = "";
    setStatus("Running the command…");

    const response = await fetch("/api/voice", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transcript })
    });
    const data = await response.json();

    if (!response.ok) {
        throw new Error(data.error);
    }

    saidBox.textContent = `Librarian: “${data.speech}”`;
    showTrace(data.trace);
    loadItems();                       // show the change in the list right away

    setStatus("Speaking…");
    try {
        await speak(data.speech);
    } catch (error) {
        // The result is still on screen even if the audio fails
        setStatus("✗ " + error.message, true);
        return;
    }

    setStatus(IDLE_MESSAGE);
}


micButton.addEventListener("click", async () => {
    if (busy) {
        return;
    }

    unlockAudio();
    setBusy(true, true);
    heardBox.textContent = "";
    saidBox.textContent = "";
    traceBox.textContent = "";
    setStatus("🎙️ Listening… say your command.");

    try {
        const transcript = await listenOnce();
        setBusy(true, false);

        if (!transcript) {
            setStatus("I didn't hear anything. Click the microphone and try again.", true);
            return;
        }

        await runTranscript(transcript);
    } catch (error) {
        setStatus("✗ " + error.message, true);
    } finally {
        setBusy(false);
    }
});


// Typed command: same pipeline, no microphone
commandForm.addEventListener("submit", async (event) => {
    event.preventDefault();

    const transcript = commandInput.value.trim();
    if (!transcript || busy) {
        return;
    }

    unlockAudio();
    setBusy(true);

    try {
        await runTranscript(transcript);
        commandInput.value = "";
    } catch (error) {
        setStatus("✗ " + error.message, true);
    } finally {
        setBusy(false);
    }
});