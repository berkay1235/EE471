const display = document.getElementById("display");
const expressionBox = document.getElementById("expression");
const errorBox = document.getElementById("error");
const micButton = document.getElementById("mic");
const voiceStatus = document.getElementById("voiceStatus");
const heardBox = document.getElementById("heard");
const calcButtons = document.querySelectorAll(".buttons button");

// =====================================================================
// 1) Calculator state (Week 4 logic, extended)
// =====================================================================
let firstValue = "";
let secondValue = "";
let operation = null;
let enteringSecondValue = false;
let justEvaluated = false;   // after "=", a new digit starts a new calculation

const SYMBOL = { "/": "÷", "*": "×", "-": "−", "+": "+" };

// What the calculator says for each button (command table of the assignment)
const WORD = {
    "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
    "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine",
    ".": "Point", "/": "Over", "*": "Times", "+": "Plus", "-": "Minus"
};

function updateDisplay(value) {
    display.textContent = value;
}

function updateExpression() {
    expressionBox.textContent = operation
        ? `${firstValue} ${SYMBOL[operation]} ${secondValue}`
        : "";
}

function clearError() {
    errorBox.textContent = "";
}

function formatNumber(n) {
    // 0.1 + 0.2 -> 0.3 instead of 0.30000000000000004
    return String(Number(Number(n).toPrecision(12)));
}

function resetCalculator() {
    firstValue = "";
    secondValue = "";
    operation = null;
    enteringSecondValue = false;
    justEvaluated = false;
    updateDisplay("0");
    updateExpression();
}

// =====================================================================
// 2) One entry point for every button: mouse, keyboard and voice
//    silent = true when the press came from a voice command
// =====================================================================
async function press(value, { silent = false } = {}) {
    clearError();

    if (/^[0-9.]$/.test(value)) {
        inputDigit(value, silent);
    } else if (value === "=") {
        await evaluate(silent);
    } else if (SYMBOL[value]) {
        await chooseOperator(value, silent);
    }
}

function inputDigit(value, silent) {
    if (justEvaluated) resetCalculator();          // start fresh after a result

    let current = enteringSecondValue ? secondValue : firstValue;
    if (value === "." && current.includes(".")) return;
    if (value === "." && current === "") current = "0";
    current += value;

    if (enteringSecondValue) secondValue = current;
    else firstValue = current;

    updateDisplay(current);
    updateExpression();
    if (!silent) say(WORD[value]);
}

async function chooseOperator(value, silent) {
    if (firstValue === "") {
        errorBox.textContent = "Enter the first number.";
        return;
    }

    // Chaining: "5 + 2 +" first calculates 7, then continues with "7 +"
    if (enteringSecondValue && secondValue !== "") {
        if (await calculate() === null) return;
    }

    operation = value;
    enteringSecondValue = true;
    justEvaluated = false;
    updateExpression();
    if (!silent) say(WORD[value]);
}

async function evaluate(silent) {
    if (firstValue === "" || secondValue === "" || operation === null) {
        errorBox.textContent = "Invalid input.";
        return;
    }

    const expression = `${firstValue} ${SYMBOL[operation]} ${secondValue} =`;
    const result = await calculate();
    if (result === null) return;

    expressionBox.textContent = expression;
    justEvaluated = true;
    if (!silent) say("is " + result);               // "is eight"
}

// Sends the calculation to Flask (same /api/calculate as Week 4)
async function calculate() {
    try {
        const response = await fetch("/api/calculate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ x: firstValue, y: secondValue, operation: operation })
        });
        const data = await response.json();

        if (!response.ok) {
            errorBox.textContent = data.error;
            return null;
        }

        const result = formatNumber(data.result);
        updateDisplay(result);
        firstValue = result;
        secondValue = "";
        operation = null;
        enteringSecondValue = false;
        return result;
    } catch (error) {
        errorBox.textContent = "Server error.";
        return null;
    }
}

// Mouse clicks
calcButtons.forEach((button) => {
    button.addEventListener("click", () => {
        unlockAudio();
        press(button.dataset.value);
    });
});

// Keyboard (Week 4 Backspace / Escape, plus digits and operators)
document.addEventListener("keydown", (event) => {
    const key = event.key;

    if (/^[0-9.+\-*/]$/.test(key)) {
        unlockAudio();
        flashButton(key);
        press(key);
    } else if (key === "Enter" || key === "=") {
        event.preventDefault();
        unlockAudio();
        flashButton("=");
        press("=");
    } else if (key === "Backspace") {
        clearError();
        if (enteringSecondValue) {
            secondValue = secondValue.slice(0, -1);
            updateDisplay(secondValue || "0");
        } else {
            firstValue = firstValue.slice(0, -1);
            updateDisplay(firstValue || "0");
        }
        updateExpression();
    } else if (key === "Escape") {
        clearError();
        resetCalculator();
    }
});

function flashButton(value) {
    const button = document.querySelector(`.buttons button[data-value="${value}"]`);
    if (!button) return;
    button.classList.add("flash");
    setTimeout(() => button.classList.remove("flash"), 250);
}

// =====================================================================
// 3) The calculator's voice (Text-to-Speech through Flask /api/speak)
//    Web Audio is used so Safari also plays sounds that arrive after a fetch.
// =====================================================================
let audioCtx = null;
const audioCache = new Map();          // text -> Promise<AudioBuffer>
let speechQueue = Promise.resolve();   // words are spoken one after another

function unlockAudio() {
    // Browsers only allow sound after a user action, so we create/resume
    // the AudioContext inside the click handler.
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        Object.values(WORD).forEach(loadSpeech);   // pre-load all words
    }
    if (audioCtx.state === "suspended") audioCtx.resume();
}

function loadSpeech(text) {
    if (!audioCache.has(text)) {
        const promise = fetch("/api/speak", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        })
            .then((r) => {
                if (!r.ok) throw new Error("Text-to-speech failed");
                return r.arrayBuffer();
            })
            .then((buffer) => audioCtx.decodeAudioData(buffer));

        promise.catch(() => audioCache.delete(text));   // retry next time
        audioCache.set(text, promise);
    }
    return audioCache.get(text);
}

function playBuffer(buffer) {
    return new Promise((resolve) => {
        const source = audioCtx.createBufferSource();
        source.buffer = buffer;
        source.connect(audioCtx.destination);
        source.onended = resolve;
        source.start();
    });
}

function say(text) {
    // While the mic is on, stay quiet: otherwise it would hear its own voice
    if (listening || !text) return;
    unlockAudio();

    const bufferPromise = loadSpeech(text);   // start downloading immediately
    speechQueue = speechQueue.then(async () => {
        try {
            await playBuffer(await bufferPromise);
        } catch (error) {
            errorBox.textContent = error.message;
        }
    });
}

// =====================================================================
// 4) Voice commands (real-time Speech-to-Text with the Azure Speech SDK)
// =====================================================================
const COMMANDS = {
    zero: "0", oh: "0", one: "1", won: "1", two: "2", to: "2", too: "2",
    three: "3", four: "4", for: "4", five: "5", six: "6", seven: "7",
    eight: "8", ate: "8", nine: "9",
    point: ".", dot: ".",
    over: "/", divide: "/", divided: "/",
    times: "*", multiply: "*", multiplied: "*", x: "*",
    plus: "+", add: "+",
    minus: "-", subtract: "-",
    is: "=", equals: "=", equal: "="
};
const SYMBOL_COMMANDS = { "+": "+", "-": "-", "−": "-", "*": "*", "×": "*", "/": "/", "÷": "/", "=": "=" };

// "five plus two is" -> ["5", "+", "2", "="]   (also understands "5 + 2 =")
function toCommands(text) {
    const tokens = text.toLowerCase().match(/\d+(?:\.\d+)?|[a-z]+|[+\-−*×/÷=]/g) || [];
    const commands = [];
    for (const token of tokens) {
        if (/^\d/.test(token)) commands.push(...token.split(""));
        else if (COMMANDS[token]) commands.push(COMMANDS[token]);
        else if (SYMBOL_COMMANDS[token]) commands.push(SYMBOL_COMMANDS[token]);
    }
    return commands;
}

let recognizer = null;
let listening = false;
let tokenTimer = null;
let commandQueue = Promise.resolve();

async function getSpeechToken() {
    const response = await fetch("/api/speech-token");
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not get a speech token");
    return data;   // { token, region }
}

function handleRecognized(result) {
    // The "Lexical" form keeps spoken words ("five plus two is") instead of "5 + 2 is"
    let lexical = "";
    try {
        const json = JSON.parse(result.properties.getProperty(SpeechSDK.PropertyId.SpeechServiceResponse_JsonResult));
        lexical = (json.NBest && json.NBest[0] && json.NBest[0].Lexical) || "";
    } catch (e) { /* use the display text below */ }

    let commands = toCommands(lexical);
    if (commands.length === 0) commands = toCommands(result.text);

    heardBox.textContent = `Heard: “${result.text}”` + (commands.length ? "" : " (no command)");

    // Execute the words in order, as if the buttons were pressed
    commandQueue = commandQueue.then(async () => {
        for (const command of commands) {
            flashButton(command);
            await press(command, { silent: true });
            await new Promise((r) => setTimeout(r, 150));
        }
    });
}

async function startListening() {
    if (!window.SpeechSDK) {
        voiceStatus.textContent = "✗ Azure Speech SDK could not be loaded (check your internet).";
        return;
    }

    voiceStatus.textContent = "Connecting...";
    const { token, region } = await getSpeechToken();

    const speechConfig = SpeechSDK.SpeechConfig.fromAuthorizationToken(token, region);
    speechConfig.speechRecognitionLanguage = "en-US";
    speechConfig.outputFormat = SpeechSDK.OutputFormat.Detailed;
    // Finish a phrase after 0.5 s of silence -> faster reaction to short commands
    speechConfig.setProperty(SpeechSDK.PropertyId.Speech_SegmentationSilenceTimeoutMs, "500");

    const audioConfig = SpeechSDK.AudioConfig.fromDefaultMicrophoneInput();
    recognizer = new SpeechSDK.SpeechRecognizer(speechConfig, audioConfig);

    // Phrase list: tells Azure which words to expect -> better accuracy for "is", "over"...
    const phraseList = SpeechSDK.PhraseListGrammar.fromRecognizer(recognizer);
    ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
     "point", "over", "times", "plus", "minus", "is"].forEach((p) => phraseList.addPhrase(p));

    recognizer.recognizing = (s, e) => {
        heardBox.textContent = "… " + e.result.text;     // live, while you speak
    };
    recognizer.recognized = (s, e) => {
        if (e.result.reason === SpeechSDK.ResultReason.RecognizedSpeech && e.result.text) {
            handleRecognized(e.result);
        }
    };
    recognizer.canceled = (s, e) => {
        if (e.reason === SpeechSDK.CancellationReason.Error) {
            voiceStatus.textContent = "✗ " + e.errorDetails;
            stopListening();
        }
    };

    recognizer.startContinuousRecognitionAsync(
        () => {
            listening = true;
            micButton.classList.add("listening");
            voiceStatus.textContent = "🎙️ Listening… say e.g. “five”, “plus”, “two”, “is”. Click the mic again to stop.";
        },
        (error) => {
            voiceStatus.textContent = "✗ " + error;
            stopListening();
        }
    );

    // Tokens expire after 10 minutes -> refresh every 9 minutes
    tokenTimer = setInterval(async () => {
        try {
            if (recognizer) recognizer.authorizationToken = (await getSpeechToken()).token;
        } catch (e) { /* next try in 9 minutes */ }
    }, 9 * 60 * 1000);
}

function stopListening() {
    clearInterval(tokenTimer);
    if (recognizer) {
        const r = recognizer;
        recognizer = null;
        r.stopContinuousRecognitionAsync(() => r.close(), () => r.close());
    }
    listening = false;
    micButton.classList.remove("listening");
    if (!voiceStatus.textContent.startsWith("✗")) {
        voiceStatus.textContent = "Microphone off. Press a button to hear it, or click 🎙️ to give voice commands.";
    }
}

micButton.addEventListener("click", async () => {
    unlockAudio();
    if (listening || recognizer) {
        stopListening();
        return;
    }
    try {
        await startListening();
    } catch (error) {
        voiceStatus.textContent = "✗ " + error.message;
        stopListening();
    }
});
