// Variables
let mic, recorder, soundFile;
const myBlobs = new Array();
const myBlobsURLs = new Array();
let main_tessitura = null;
let email = null;
let selectedSex = null;
let length_text_blobs = 0; 
let count_record = 0;
// Buttons
const recordButton = document.getElementById("recordButton");
const stopButton = document.getElementById("stopButton");
const deleteButton = document.getElementById("deleteButton");
let nextButton = document.getElementById("nextButton");
let sendButton = null;

// Elements
const infoSection = document.getElementById("body-info");
const recordSection = document.getElementById("record");
const recordingsList = document.getElementById("recordingsList");
const status = document.getElementById("status");
const emailTag = document.getElementById("email");
const sex = document.getElementById("sex");
const recordsBox = document.getElementById("records-box");
const phrasesTag = document.getElementById("phrases");
const tessituraTag = document.getElementById("tessitura-field");
const resultsTag = document.getElementById("results");
const formUserData = document.getElementById("user-data");

const options = {
    series: [],
    chart: {
        type: "bar",
        height: 350,
        stacked: true,
        stackType: "100%",
    },
    plotOptions: {
        bar: {
            horizontal: true,
        },
    },
    stroke: {
        width: 1,
        colors: ["#fff"],
    },
    title: {
        text: "Resultados de la clasificación",
    },
    xaxis: {
        categories: ["Tipo"],
    },
    tooltip: {
        y: {
            formatter: function (val) {
                return val + "%";
            },
        },
    },
    fill: {
        opacity: 1,
    },
    legend: {
        position: "top",
        horizontalAlign: "left",
        offsetX: 40,
    },
};

// Add events to buttons
recordButton.addEventListener("click", recordPressed);
stopButton.addEventListener("click", stopPressed);
nextButton.addEventListener("click", nextPressed);
deleteButton.addEventListener("click", deletePressed);

// Add event to select tag
sex.addEventListener("change", sexSelected);

function sexSelected() {
    // Defaul Tessitura Value
    main_tessitura = `Desconocido`;

    // Get selected sex
    selectedSex = sex.options[sex.selectedIndex].value;

    // Render tessitura field
    tessituraTag.innerHTML = tessituraFieldTemplate(selectedSex);

    // Record button activated with effect
    recordButton.classList.add("respirar");
    recordButton.disabled = false;

    // Render phrases
    phrasesTag.innerHTML = phrasesTemplate();

    // Config radio change event
    configRadioEvents();
}

function shufflePhrases() {
    for (let k = 0; k < 100; k++) {
        for (let i = 0; i < phrases.length; i++) {
            const element = phrases[i];
            const j = Math.floor(Math.random() * phrases.length);
            phrases[i] = phrases[j];
            phrases[j] = element;
        }
    }
}

function tessituraFieldTemplate(sex) {
    let tessituraTemplate = '<h3 class="lead">En tu experiencia<br/>¿Cuál es tu tipo de voz?</h3>';
    if (sex == "Woman") {
        tessituraTemplate += `
        <div class="form-group">
            <div class="form-check form-check-inline">
                <input class="form-check-input" type="radio" name="main_tessitura" id="tessitura1" value="Contralto">
                <label class="form-check-label" for="tessitura1">Contralto</label>
            </div>
            <div class="form-check form-check-inline">
                <input class="form-check-input" type="radio" name="main_tessitura" id="tessitura2" value="Mezzosoprano">
                <label class="form-check-label" for="tessitura2">Mezzosoprano</label>
            </div>
            <div class="form-check form-check-inline">
                <input class="form-check-input" type="radio" name="main_tessitura" id="tessitura3" value="Soprano">
                <label class="form-check-label" for="tessitura3">Soprano</label>
            </div>
            <div class="form-check form-check-inline">
                <input class="form-check-input" type="radio" name="main_tessitura" id="tessitura4" value="Desconocido" checked>
                <label class="form-check-label" for="tessitura4">Desconocido</label>
            </div>
        </div>
        `;
    } else {
        tessituraTemplate += `
        <div class="form-group">
            <div class="form-check form-check-inline">
                <input class="form-check-input" type="radio" name="main_tessitura" id="tessitura1" value="Bajo">
                <label class="form-check-label" for="tessitura1">Bajo</label>
            </div>
            <div class="form-check form-check-inline">
                <input class="form-check-input" type="radio" name="main_tessitura" id="tessitura2" value="Barítono">
                <label class="form-check-label" for="tessitura2">Barítono</label>
            </div>
            <div class="form-check form-check-inline">
                <input class="form-check-input" type="radio" name="main_tessitura" id="tessitura3" value="Tenor">
                <label class="form-check-label" for="tessitura3">Tenor</label>
            </div>
            <div class="form-check form-check-inline">
                <input class="form-check-input" type="radio" name="main_tessitura" id="tessitura4" value="Desconocido" checked>
                <label class="form-check-label" for="tessitura4">Desconocido</label>
            </div>
        </div>
        `;
    }
    return tessituraTemplate;
}

function phrasesTemplate() {
    // Shuffle phrases
    shufflePhrases();
    let phrasesTemplate = `
    <div class="alert alert-info" role="alert">
        ¡Para clasificar tu voz necesitamos que presiones <button style="cursor: default;" class="btn btn-success" disabled>Grabar</button> y leas los siguientes textos durante la grabación!
    </div>
    `;
    for (let i = 0; i < 3; i++) {
        phrasesTemplate += `<p>${phrases[i]}</p><hr />`;
    }
    return phrasesTemplate;
}

function configRadioEvents() {
    const radios = document.getElementsByName(`main_tessitura`);
    for (let i = 0; i < radios.length; i++) {
        radios[i].addEventListener("change", function () {
            if (this.value !== main_tessitura) {
                main_tessitura = this.value;
            }             
        });
    }
}

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
    if ((!sendButton && myBlobs.length < 1)
    ||
    (sendButton && myBlobs.length < 2) ) {
    // Config classes for buttons
    recordButton.classList.remove("respirar");
    stopButton.classList.add("respirar");
    // Change buttons status
    nextButton.disabled = true;
    stopButton.disabled = false;
    recordButton.disabled = true;
    // Show recording status
    status.innerHTML = "Grabando...";
    // Use the '.enabled' boolean to make sure user enabled the mic (otherwise we'd record silence)
    if (mic.enabled) {
        // Tell recorder to record to a p5.SoundFile which we will use for playback
        recorder.record(soundFile);
    }
    } else {
        alert("No puedes grabar más en esta sección");
    }
}

function stopPressed() {
    // Clean status text
    status.innerHTML = "";
    // Stop recorder
    recorder.stop();
    // Call storeData method
    storeData();
    // Render records box
    recordsBox.classList.remove("d-none");
    // Change buttons status
    nextButton.disabled = false;
    stopButton.disabled = true;    
    recordButton.disabled = true;
    recordButton.classList.remove("respirar");
    
    
    if (!sendButton) {
        nextButton.classList.add("respirar");
    } else {
        sendButton.classList.add("respirar");
    }    
    stopButton.classList.remove("respirar");
}

function storeData() {
    // Create URL with blob object
    const url = URL.createObjectURL(soundFile.getBlob());
    // Create <audio> element
    const audioTag = document.createElement("audio");
    // Create <li> element
    const li = document.createElement("li");
    // Append blob and URL to arrays
    myBlobs.push(soundFile.getBlob());
    myBlobsURLs.push(url);
    // Add controls to the <audio> element
    audioTag.controls = true;
    // Add URL to de <audio> element
    audioTag.src = url;
    // Add class to the li element
    li.classList.add("list-group-item");
    // Add the new audio elements to the li element
    li.appendChild(audioTag);
    // Add the li element to the ordered list
    recordingsList.appendChild(li);
}

function deletePressed() {
    // if there are blobs
    if (myBlobs.length) {
        // remove the last blob and the last URL from arrays
        if(recordingsList.hasChildNodes())
        recordingsList.removeChild(recordingsList.lastChild);
        myBlobs.pop();
        myBlobsURLs.pop();
        // if there are no blobs
        if (!myBlobs.length) {
            // Hide records box
            recordsBox.classList.add("d-none");
        }
    }    
        recordButton.disabled = false;
        recordButton.classList.add("respirar");
        // Render records box
        recordsBox.classList.add("d-none");
    
}
function nextPressed() {

    // Config classes for send button
    nextButton.classList.remove("respirar");
    // Get full name
    if(!email)
    email = emailTag.value;
    // Validate form data
    if (!email || !main_tessitura || (selectedSex != "Man" && selectedSex != "Woman") || myBlobs.length<1) {
        emailTag.classList.add("border", "border-danger");
        alert("Faltan datos por ingresar");
        return;
    } else {
        emailTag.classList.remove("border", "border-danger");
    }

    getGlisandoTemplate();
    let newButton = document.createElement("Button"); 
    newButton.innerHTML = "Enviar"; 
    newButton.classList.add("btn", "btn-primary");
    newButton.id = "sendButton";
   
    nextButton.parentNode.insertBefore(newButton, nextButton);
    nextButton.parentNode.removeChild(nextButton);

    sendButton = document.getElementById("sendButton");

    sendButton.addEventListener("click", sendPressed);
}

function sendPressed() {
    
    // Build form data to send
    const config = buildFormData();
   // Show spinner
   resultsTag.innerHTML = ` <div class="spinner-border text-success" role="status">
                                <span class="sr-only">Loading...</span>
                            </div>`;
    // Send data to server    
    fetch("/classifications/classify", config)
        .then(function (response) {
            // Return JSON response
            return response.json();
        }).then(function (results) {
           // console.log(results);   
            if(results[1]["Warning"] == "True"){
                resultsTag.innerHTML = "";
                getRecordAgainTemplate();
                return;
            }
            // Erase Section
            infoSection.parentNode.removeChild(infoSection);
            
            // Build results template
            let resultTemplate = `<h4>Resultados de la clasificación</h4>
                                    <h3>Tu registro vocal es: 
                                    <strong>
                                    ${results[1]["Tessitura"][0]}
                                    </strong></h3>`;       

            resultTemplate += getResultTemplate(results);
            // Render results
            resultsTag.innerHTML = resultTemplate;
        
            //Generar notas en el piano
            const notes = generateNotes(6);
            const indexnotes = generateNotesIndex(notes);
            const objs = generateObjects(results);
            for (const key in objs) {
                generateStyle(objs[key],notes,indexnotes);
            } 
            //Config checkboxes events
            configCheckboxEvents();           
            // Plots
            plotResultsCharts(results);
        })
        .catch(() => {
            resultsTag.innerHTML = `<div class="alert alert-danger text-left" role="alert">
                        <ul>
                            <li>Verifique la grabación que acaba de realizar</li>
                            <li>Verifique que está a la distancia correcta del micrófono</li>
                        </ul>
                    </div>`;
        });  
}

function buildFormData() {
    if (myBlobs.length > 0) {        
        // Create a new FormData object
        const fd = new FormData(formUserData);
        // Append blobs to FormData object
        for (let blob = 0; blob < length_text_blobs; blob++) {
            const element = myBlobs[blob];
            fd.append(
                "audio_data_" + blob,
                element,
                `${email.replace(/@/gi, "_")}-${selectedSex}-${main_tessitura}-${blob}_${now()}.wav`
            );
        }        
        fd.append(
            "audio_data_glisando",
            myBlobs[length_text_blobs],
            `${email.replace(/@/gi, "_")}-${selectedSex}-${main_tessitura}-glisando_${now()}.wav`
        );
        fd.append(
            "count_times",
            count_record
        );
        // Get csrf token from cookies
        const csrftoken = getCookie("csrftoken");
        // Config headers
        const headers = new Headers({
            "X-CSRFToken": csrftoken,
        });
        // Config the request
        const config = {
            method: "POST",
            headers: headers,
            body: fd,
            //mode: 'cors',
            cache: "default",
        };
        return config;
    }
}

function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== "") {
        const cookies = document.cookie.split(";");
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === name + "=") {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

function now() {
    const date = new Date();
    return `${date.getFullYear()}-${date.getMonth() + 1}-${date.getDate()}_${date.getHours()}-${date.getMinutes()}}`;
}

function getResultTemplate(result) {
    let resultTemplate = `<h5>
                        Resultado del glissando 
                    </h5>
                    <hr>
                    <div class="row">                    
                    `;
     
    for (const key in result[0]["AllTessitura"]) {
        resultTemplate += ` <div class="custom-control custom-checkbox col-6 col-md-3">
                                <input type="checkbox" name="tessitura" class="custom-control-input" id="ch-${result[0]["AllTessitura"][key]}" checked>
                                <label class="custom-control-label" for="ch-${result[0]["AllTessitura"][key]}">${result[0]["AllTessitura"][key]}</label>
                            </div>`;
    }

    resultTemplate += `
                        <div class="custom-control custom-checkbox col-6 col-md-3">
                            <input type="checkbox" name="tessitura" class="custom-control-input" id="ch-tuVoz" checked>
                            <label class="custom-control-label" for="ch-tuVoz">Tu voz</label>
                        </div>
                    </div>
                    <div class = "x-scroller">
                        <div class="divPiano">
                            <div class="custom-control custom-switch switch_notas">
                                <input type="checkbox" class="custom-control-input" id="notasLatin">
                                <label class="custom-control-label" for="notasLatin">Notas en Lat&iacute;n</label>
                            </div> `;
    for (const key in result[0]["AllTessitura"]) {
        resultTemplate += `
                        <div id="${result[0]["AllTessitura"][key]}-parent" class="div_tess">
                            <div id="${result[0]["AllTessitura"][key]}" class="scala">
                            </div>
                           <div id="${result[0]["AllTessitura"][key]}-limits" class="limits">
                           </div>
                        </div>`;
    }
    resultTemplate += `
                        <div id="tuVoz-parent" class="div_tess">
                            <div id="tuVoz" class="scala">
                                </div>
                                <div id="tuVoz-limits" class="limits">
                                </div>
                        </div>
                            <img class="img-piano" src="${pianoPath}" alt="Piano"> 
                            </div>
                    </div>
                    <div class="card">
                        <div class="card-header">
                            <a class="card-link" data-toggle="collapse" href="#collapseInfo">
                                Más información
                            </a>
                        </div>
                    <div id="collapseInfo" class="collapse">
                        <div class="card-body">
                            <div id="chart">
                                </div>
                                <hr/><table class="table table-striped text-left">
                                <thead>
                                    <tr>
                                        <th scope="col"></th>
                                        <th scope="col">Nota</th>
                                        <th scope="col">Nota en Lat&iacute;n</th>
                                    </tr>
                                </thead>
                                <tbody>
                                `;

    resultTemplate += `<tr><td> La nota más baja que detectamos es </td> 
                            <td>${result[0]["ResultNotes"]["MinMaxNote"][0]}</td>
                            <td>${result[0]["ResultNotes"]["MinMaxLatin"][0]}</td>
                        </tr>`;      
            
    resultTemplate += ` <tr><td> La nota más alta que detectamos es </td> 
                            <td>${result[0]["ResultNotes"]["MinMaxNote"][1]}</td>
                            <td>${result[0]["ResultNotes"]["MinMaxLatin"][1]}</td>
                        </tr>`;
     
    resultTemplate += `             </tbody>
                                </table>
                            </div>
                        </div>
                    </div>`;


    resultTemplate += `
    <a href="/classifications/record" class="btn btn-primary mt-5">Volver a realizar</a>
    `   

    return resultTemplate;
}

function plotResultsCharts(result) {    
    for (const key in result[0]["AllTessitura"]) {
        options.series[key] = { name: result[0]["AllTessitura"][key], data: [result[1]["Result"][key]] };
    }    
    if (options.series[0].name !== "No definida" && options.series[1].name !== "No definida" && options.series[0].name !== "No definida") {
        const chart = new ApexCharts(document.querySelector(`#chart`), options);
        chart.render();
    }
}

function generateNotes(octavas){
    let basic_notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    let complete_notes = basic_notes.slice();
    
    for (let index = 0; index < octavas; index++) {    
        for (let j = 0; j < basic_notes.length; j++) {    
        	let note = `${basic_notes[j]}${index+1}`;
            complete_notes.push(note);
        }   
    }
    return complete_notes;
}

function generateNotesIndex(notes){
    let index_notes = [];
    let count_white = 0, 
        count_black = 0;
    for (i=0; i<notes.length; i++) {
        if (notes[i].includes("#")) {
            count_black++;
            index_notes.push(count_black);
        } else {
            count_white++;
            index_notes.push(count_white);    
        }        
    }
    return index_notes;
}

function searchRangeNotesIndex(notes,range){
	return [notes.indexOf(range[0]), notes.indexOf(range[1])];
}

function generateObjects(results) {
    let objs = [],
    tessituras=['Soprano', 'Mezzosoprano', 'Contralto',
                'Tenor','Barítono','Bajo'],
    ranges=[
        ['F3', 'D6'], 
        ['D#3', 'B5'],
        ['C3', 'G5'],               
        ['G2', 'B4'],
        ['E2', 'G4'],
        ['C2', 'E4'],
    ],
    ranges_latin=[
        ['Fa3', 'Re6'], 
        ['D#3', 'B5'],
        ['Do3', 'Sol5'],               
        ['Sol2', 'Si4'],
        ['Mi2', 'Sol4'],
        ['Do2', 'Mi4'],
    ]
    colors = ['#8e44ad', '#2980b9', '#e74c3c', '#2ecc71'];
    for (const key in results[0]["AllTessitura"]) {
        let obj = {
            id: key,
            tipo: results[0]["AllTessitura"][key],
            range: ranges[tessituras.indexOf(results[0]["AllTessitura"][key])], 
            rangeLatin: ranges_latin[tessituras.indexOf(results[0]["AllTessitura"][key])] ,
            color: colors[key],
        };       
        objs.push(obj);
    }
    let obj = {
        id: 3,
        tipo: "tuVoz",
        range: results[0]["ResultNotes"]["MinMaxNote"], 
        rangeLatin: results[0]["ResultNotes"]["MinMaxLatin"],
        color: colors[3],
    };       
    objs.push(obj);
    return objs;
}

function generateStyle(obj,n,n_i){
    let range = searchRangeNotesIndex(n,obj.range);
    let black_left = [7.9,9.9,13.3,15.3,17.3,20.5,22.5,26,28,30,33.2,35.2,38.6,40.6,42.4,45.7,47.7,51.2,53.2,55.1,58.4,60.4,53.8,65.7,67.7,70.9,72.9,76.4,78.4,80.4,83.6,85.6,88.9,90.9,92.9];
    
    let left_i = n[range[0]].includes("#") ?
    black_left[n_i[range[0]]-1] :
    6.75 + 1.8 * (n_i[range[0]] - 1),
    left_f = n[range[1]].includes("#") ?
    black_left[n_i[range[1]]-1] :
    6.75 + 1.8 * (n_i[range[1]] - 1),
    w = n[range[1]].includes("#") ?
    ((left_f + 1) - 6.75) - (left_i - 6.75) :
    (1.8 * n_i[range[1]]) - (left_i - 6.75),
    left = left_i.toString() + '%',
    width = w.toString() + '%';4
    
    const style = obj.tipo!="tuVoz"? 
                `               
                background-color: ${obj.color};                
                `:
                `               
                height: 67%;                             
                background-color: ${obj.color};                
                `;
    let limit_style =
                `            
                border-left: 0.3em solid ${obj.color};
                border-right: 0.3em solid ${obj.color};
                color: ${obj.color}; 
                font-size: 1rem;               
                `,
    parent_style= obj.tipo!="tuVoz"? 
                `
                width: ${width};
                left: ${left};
                top: ${(45+13.5*obj.id).toString()}%;                            
                `:
                `
                width: ${width};
                left: ${left};      
                top: 45%;                
                height: 60%;                     
                                
    `;

    document.getElementById(obj.tipo).setAttribute("style",style);  
    document.getElementById(`${obj.tipo}-parent`).setAttribute("style",parent_style);      
    const limits = document.getElementById(`${obj.tipo}-limits`),
    s_limit = obj.tipo!="tuVoz"? `0.5em` : '-2.1em',
    s_limit_latin = obj.tipo!="tuVoz"? `0.5em` : '-3em',
    s_name = obj.tipo!="tuVoz"? `${obj.tipo}` : 'Tu Voz',
    span_limits = `style="
                position: absolute;
                bottom: 0%;
                `,
    span_limits_t = `style="
                position: relative;
                top: 89%;
                `;

    limits.setAttribute("style",limit_style);    
    limits.innerHTML = `
                <span class="nota_normal" ${span_limits} left:${s_limit};"> ${obj.range[0]} </span>
                <span class="nota_latin hidden_note" ${span_limits} left:${s_limit_latin};"> ${obj.rangeLatin[0]} </span>
                <span ${span_limits_t}"> ${s_name} </span>
                <span class="nota_normal"${span_limits} right:${s_limit};"> ${obj.range[1]} </span>
                <span class="nota_latin hidden_note"${span_limits} right:${s_limit_latin};"> ${obj.rangeLatin[1]} </span>
    `;
 }
function configCheckboxEvents(){
    const checkboxes= document.getElementsByName(`tessitura`);
    for (let i = 0; i < checkboxes.length; i++) {
            checkboxes[i].addEventListener("change", function () {
                const id=checkboxes[i].id.substring(3, 16),
                element = document.getElementById(`${id}-parent`);                            
                if (checkboxes[i].checked)
                {   
                    if(element.id!="tuVoz-parent") {
                        element.classList.add("tessitura_in");   
                        element.classList.remove("tessitura_out");  
                    } else {
                        element.classList.add("voice_in");   
                        element.classList.remove("voice_out");  
                    }                                        
                } else {
                    if(element.id!="tuVoz-parent") {
                        element.classList.add("tessitura_out");   
                        element.classList.remove("tessitura_in");  
                    } else {
                        element.classList.add("voice_out");   
                        element.classList.remove("voice_in");  
                    }                    
                }               
            });
        }
        switch_notas = document.getElementById(`notasLatin`); 
        switch_notas.addEventListener("change", function () {
            if (switch_notas.checked){
                $('.nota_normal').addClass("hidden_note");
                $('.nota_latin').removeClass("hidden_note");
            } else {
                $('.nota_latin').addClass("hidden_note");
                $('.nota_normal').removeClass("hidden_note");
            }
        });

}
function getRecordAgainTemplate(){

    while(myBlobs.length>0){
        deletePressed();
    }
    let recordText =
    `
    <div class="card alert alert-danger shadow-lg my-3">
        <div class="card-body ">
        El registro alcanzado en el glissando no corresponden con su voz natural, 
        por favor repita la prueba y tenga en cuenta lo siguiente:
            <br /><strong>
            <ul class="text-left">                            
                <li>Realice el glissando desde su nota m&aacute;s grave hasta su nota m&aacute;s aguda.
                <small><strong>Nota: </strong> No importan las notas intermedias.</small></li>
                <li>No finja la voz.</li>
                <li>No fuerce la voz.</li>               
            `;
       recordText += selectedSex=='Man'? "<li>Intente no hacer falsete.</li>":"";      
       recordText +=        
            `</ul>
            Nota: Por favor verifique la grabación antes de enviarla
            </strong>
        </div>
    </div>
    <div id="phrases"></div>
    `;
    recordSection.innerHTML = recordText;
    let newButton = document.createElement("Button"); 
    newButton.innerHTML = "Siguente"; 
    newButton.classList.add("btn", "btn-primary");
    newButton.id = "nextButton";
   
    sendButton.parentNode.insertBefore(newButton, sendButton);
    sendButton.parentNode.removeChild(sendButton);

    nextButton = document.getElementById("nextButton");
    nextButton.classList.add("respirar");
    nextButton.addEventListener("click", nextPressed);
     // Record button activated with effect
     recordButton.classList.add("respirar");
     recordButton.disabled = false;
     const phrasesTag2 = document.getElementById("phrases");
     // Render phrases
     phrasesTag2.innerHTML = phrasesTemplate();
     //Incrementamos por 2da grabación
     count_record++;
     window.scrollTo(0, 0);
}
function getGlisandoTemplate(){ 
    
    length_text_blobs = myBlobs.length;

    for (let index = 0; index <  length_text_blobs; index++) {       
        // remove the last blob and the last URL from arrays
        recordingsList.removeChild(recordingsList.lastChild);            
    }
    // Hide records box
    recordsBox.classList.add("d-none");
    recordButton.disabled = false;
    recordSection.innerHTML = `
            <h2>Glissando</h2>
            <video width="400"  controls>
                <source src="${videoPath}" type="video/mp4">
                <source src="movie.ogg" type="video/ogg">
                Your browser does not support the video tag.
            </video>           
            <div class="alert alert-info" role="alert">
                ¡Para clasificar tu voz necesitamos que presiones <button style="cursor: default;" class="btn btn-success" disabled>Grabar</button> y realices el ejercicio durante la grabación!
            </div>
            <hr />           
    `;    
    
    window.scrollTo(0, 0);
}