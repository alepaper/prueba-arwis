// Variables
let ws;
let mic, recorder, soundFile;
let recordFlag;

// Buttons
const recordButton = document.getElementById("recordButton");
const stopButton = document.getElementById("stopButton");

// Elements
const resultsTag = document.getElementById("results");
const status = document.getElementById("status");
const time = document.getElementById("time");
// Add events to buttons
recordButton.addEventListener("click", recordPressed);
stopButton.addEventListener("click", stopPressed);

function setup() {
    noCanvas();
    // Create an audio in
    mic = new p5.AudioIn();
    // Users must manually enable their browser microphone for recording to work properly!
    mic.start();
    // Create a sound recorder
    recorder = new p5.SoundRecorder();
    // Connect the mic to the recorder
    recorder.setInput(mic);
    // Create an empty sound file that we will use to playback the recording
    soundFile = new p5.SoundFile();
}

function touchStarted() {
    // update AudioContext
    getAudioContext().resume();
}

function recordPressed() {
    // Open wWebSocket channel.
    ws = new WebSocket("ws://localhost:8000/ws/");
    ws.addEventListener("message", updateNote);

    // Show recording status
    status.innerHTML = "Grabando...";
    recordFlag = true;
    record();
}

function updateNote(event) {
    let divNote = document.getElementById("note");
    if (event.data != "") {
        divNote.innerHTML = event.data;
    }
    console.log("Message from server ", event.data);
}

function record() {
    // Use the '.enabled' boolean to make sure user enabled the mic (otherwise we'd record silence)
    if (mic.enabled && recordFlag) {
        // Create a new sound file
        soundFile = new p5.SoundFile();
        // Tell recorder to record to a p5.SoundFile which we will use for playback
        recorder.record(soundFile);
        // Asincronous call to send data by slices.
        setTimeout(send, time.value);
    } else {
        alert("No puedes grabar en esta sección. El micrófono no está habilitado.");
    }
}

function send() {
    recorder.stop();

    const audioSlice = soundFile.getBlob();

    if (audioSlice) {
        ws.send(audioSlice);
    }

    record();
}

function stopPressed() {
    ws.send("-1");
    ws.close();
    // Clean status text
    status.innerHTML = "";
    recordFlag = false;
    // Stop recorder
    if (recorder) {
        recorder.stop();
    }

    // Call storeData method
}
