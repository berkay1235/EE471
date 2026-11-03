import io
import os
import tempfile
import threading

import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_file

# --- 1) Load secret keys from .env -------------------------------------------
load_dotenv()
SPEECH_KEY = os.getenv("SPEECH_KEY")
SPEECH_REGION = os.getenv("SPEECH_REGION")

app = Flask(__name__)


def make_speech_config():
    """Create an Azure SpeechConfig from the values in .env."""
    if not SPEECH_KEY or not SPEECH_REGION:
        raise RuntimeError("SPEECH_KEY / SPEECH_REGION is missing in .env")
    return speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)


def make_translation_config():
    """Translation needs its own config type."""
    if not SPEECH_KEY or not SPEECH_REGION:
        raise RuntimeError("SPEECH_KEY / SPEECH_REGION is missing in .env")
    return speechsdk.translation.SpeechTranslationConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)


# --- 2) Shared helpers -------------------------------------------------------
def save_uploaded_wav():
    """Save the 'audio' file from the request to a temp .wav and return its path.
    The browser already converted it to 16 kHz mono WAV."""
    audio = request.files.get("audio")
    if audio is None:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    audio.save(tmp)
    tmp.close()
    return tmp.name


def run_continuous(recognizer, result_event, on_result, start, stop, timeout=180):
    """Run any Azure recognizer until the audio file ends.
    - result_event: the event that fires for each recognized sentence
    - on_result:    our callback that stores the sentence
    - start/stop:   functions that start/stop this recognizer type"""
    errors = []
    done = threading.Event()

    def on_canceled(evt):
        details = evt.cancellation_details
        if details.reason == speechsdk.CancellationReason.Error:
            errors.append(details.error_details)
        done.set()  # EndOfStream also lands here when the file is finished

    result_event.connect(on_result)
    recognizer.canceled.connect(on_canceled)
    recognizer.session_stopped.connect(lambda evt: done.set())

    start()
    done.wait(timeout=timeout)
    stop()

    if errors:
        raise RuntimeError(errors[0])


# --- 3) Home page ------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


# --- 4) Text-to-Speech -------------------------------------------------------
@app.route("/synthesize", methods=["POST"])
def synthesize():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    voice = data.get("voice") or "en-US-JennyNeural"

    if not text:
        return jsonify(error="Please enter some text."), 400

    try:
        speech_config = make_speech_config()
    except RuntimeError as e:
        return jsonify(error=str(e)), 500

    speech_config.speech_synthesis_voice_name = voice
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm  # .wav
    )

    # audio_config=None -> keep audio in memory instead of playing on server speakers
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)
    result = synthesizer.speak_text_async(text).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return send_file(
            io.BytesIO(result.audio_data),
            mimetype="audio/wav",
            as_attachment=False,
            download_name="speech.wav",
        )

    details = result.cancellation_details
    return jsonify(error=f"Synthesis failed: {details.reason} {details.error_details}"), 500


# --- 5) Speech-to-Text -------------------------------------------------------
@app.route("/transcribe", methods=["POST"])
def transcribe():
    wav_path = save_uploaded_wav()
    if wav_path is None:
        return jsonify(error="Please record or upload an audio file."), 400

    try:
        speech_config = make_speech_config()
        speech_config.speech_recognition_language = request.form.get("language") or "en-US"
        audio_config = speechsdk.audio.AudioConfig(filename=wav_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        pieces = []

        def on_recognized(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech and evt.result.text:
                pieces.append(evt.result.text)

        run_continuous(recognizer, recognizer.recognized, on_recognized,
                       recognizer.start_continuous_recognition,
                       recognizer.stop_continuous_recognition)
    except RuntimeError as e:
        return jsonify(error=f"Recognition failed: {e}"), 500
    finally:
        os.remove(wav_path)

    if not pieces:
        return jsonify(error="No speech could be recognized."), 422
    return jsonify(text=" ".join(pieces))


# --- 6) Real-time diarization: who said what ---------------------------------
@app.route("/diarize", methods=["POST"])
def diarize():
    wav_path = save_uploaded_wav()
    if wav_path is None:
        return jsonify(error="Please record or upload an audio file."), 400

    try:
        speech_config = make_speech_config()
        speech_config.speech_recognition_language = request.form.get("language") or "en-US"
        audio_config = speechsdk.audio.AudioConfig(filename=wav_path)
        # ConversationTranscriber = Azure's real-time diarization API
        transcriber = speechsdk.transcription.ConversationTranscriber(
            speech_config=speech_config, audio_config=audio_config
        )

        segments = []

        def on_transcribed(evt):
            if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech and evt.result.text:
                segments.append({"speaker": evt.result.speaker_id or "Unknown",
                                 "text": evt.result.text})

        run_continuous(transcriber, transcriber.transcribed, on_transcribed,
                       lambda: transcriber.start_transcribing_async().get(),
                       lambda: transcriber.stop_transcribing_async().get())
    except RuntimeError as e:
        return jsonify(error=f"Diarization failed: {e}"), 500
    finally:
        os.remove(wav_path)

    if not segments:
        return jsonify(error="No speech could be recognized."), 422
    return jsonify(segments=segments)


# --- 7) Speech translation ---------------------------------------------------
@app.route("/translate", methods=["POST"])
def translate():
    wav_path = save_uploaded_wav()
    if wav_path is None:
        return jsonify(error="Please record or upload an audio file."), 400

    source = request.form.get("source") or "tr-TR"   # spoken language, e.g. tr-TR
    target = request.form.get("target") or "en"      # target language, e.g. en

    try:
        translation_config = make_translation_config()
        translation_config.speech_recognition_language = source
        translation_config.add_target_language(target)
        audio_config = speechsdk.audio.AudioConfig(filename=wav_path)
        recognizer = speechsdk.translation.TranslationRecognizer(
            translation_config=translation_config, audio_config=audio_config
        )

        originals, translations = [], []

        def on_recognized(evt):
            if evt.result.reason == speechsdk.ResultReason.TranslatedSpeech and evt.result.text:
                originals.append(evt.result.text)
                translations.append(evt.result.translations.get(target, ""))

        run_continuous(recognizer, recognizer.recognized, on_recognized,
                       recognizer.start_continuous_recognition,
                       recognizer.stop_continuous_recognition)
    except RuntimeError as e:
        return jsonify(error=f"Translation failed: {e}"), 500
    finally:
        os.remove(wav_path)

    if not originals:
        return jsonify(error="No speech could be recognized."), 422
    return jsonify(original=" ".join(originals), translation=" ".join(translations))


# --- 8) Pronunciation assessment ---------------------------------------------
@app.route("/pronunciation", methods=["POST"])
def pronunciation():
    wav_path = save_uploaded_wav()
    reference = (request.form.get("reference") or "").strip()
    if wav_path is None:
        return jsonify(error="Please record or upload an audio file."), 400
    if not reference:
        os.remove(wav_path)
        return jsonify(error="Please enter the reference sentence."), 400

    language = request.form.get("language") or "en-US"

    try:
        speech_config = make_speech_config()
        speech_config.speech_recognition_language = language
        audio_config = speechsdk.audio.AudioConfig(filename=wav_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        pa_config = speechsdk.PronunciationAssessmentConfig(
            reference_text=reference,
            grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
            granularity=speechsdk.PronunciationAssessmentGranularity.Word,
            enable_miscue=True,  # detect omitted / inserted words
        )
        if language == "en-US":
            pa_config.enable_prosody_assessment()  # intonation & rhythm score
        pa_config.apply_to(recognizer)

        result = recognizer.recognize_once_async().get()
    except RuntimeError as e:
        os.remove(wav_path)
        return jsonify(error=str(e)), 500

    os.remove(wav_path)

    if result.reason == speechsdk.ResultReason.NoMatch:
        return jsonify(error="No speech could be recognized."), 422
    if result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        return jsonify(error=f"Assessment failed: {details.error_details}"), 500

    pa = speechsdk.PronunciationAssessmentResult(result)
    words = [{"word": w.word,
              "score": w.accuracy_score,
              "error": w.error_type}          # None / Mispronunciation / Omission / Insertion
             for w in pa.words]

    return jsonify(
        recognized=result.text,
        scores={
            "Pronunciation": pa.pronunciation_score,
            "Accuracy": pa.accuracy_score,
            "Fluency": pa.fluency_score,
            "Completeness": pa.completeness_score,
            "Prosody": getattr(pa, "prosody_score", None),
        },
        words=words,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)  # 5000 is used by AirPlay on macOS
