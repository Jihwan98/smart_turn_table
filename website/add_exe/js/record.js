const record = document.getElementById("record")
const stop = document.getElementById("stop")
const soundClips = document.getElementById("sound-clips")


const audioCtx = new(window.AudioContext || window.webkitAudioContext)() // 오디오 컨텍스트 정의

const analyser = audioCtx.createAnalyser()
//        const distortion = audioCtx.createWaveShaper()
//        const gainNode = audioCtx.createGain()
//        const biquadFilter = audioCtx.createBiquadFilter()

function makeSound(stream) {
    const source = audioCtx.createMediaStreamSource(stream)

    source.connect(analyser)
    //            analyser.connect(distortion)
    //            distortion.connect(biquadFilter)
    //            biquadFilter.connect(gainNode)
    //            gainNode.connect(audioCtx.destination) // connecting the different audio graph nodes together
    analyser.connect(audioCtx.destination)

}
record.onclick = () => {
    record.onclick = null;
    if (navigator.mediaDevices) {
        console.log('getUserMedia supported.')
        const constraints = {
            audio: true
        }
        
        let chunks = []

        navigator.mediaDevices.getUserMedia(constraints)
            .then(stream => {
                alert("Mic allowed")
                const mediaRecorder = new MediaRecorder(stream)
                                
                record.onclick = () => {
                    mediaRecorder.start()
                    console.log(mediaRecorder.state)
                    console.log("recorder started")
                    record.style.background = "red"
                    record.style.color = "black"
                }

                stop.onclick = () => {
                    mediaRecorder.stop()
                    console.log(mediaRecorder.state)
                    console.log("recorder stopped")
                    record.style.background = ""
                    record.style.color = ""
                }

                mediaRecorder.onstop = e => {
                    console.log("data available after MediaRecorder.stop() called.")

                    const clipName = new Date()

                    const clipContainer = document.createElement('article')
                    const clipLabel = document.createElement('span')
                    const audio = document.createElement('audio')
                    const deleteButton = document.createElement('button')
                    const br = document.createElement('br')

                    clipContainer.classList.add('clip')
                    deleteButton.classList.add('delete_btn')
                    audio.setAttribute('controls', '')
                    deleteButton.innerHTML = "삭제"
                    // clipLabel.innerHTML = clipName

                    clipContainer.appendChild(audio)
                    clipContainer.appendChild(clipLabel)
                    clipContainer.appendChild(deleteButton)
                    soundClips.appendChild(clipContainer)
                    // clipContainer.append(br)

                    audio.controls = true
                    const blob = new Blob(chunks, {
                        'type': 'audio/ogg codecs=opus'
                    })
                    chunks = []
                    const audioURL = URL.createObjectURL(blob)
                    audio.src = audioURL
                    console.log("recorder stopped")

                    deleteButton.onclick = e => {
                        evtTgt = e.target
                        evtTgt.parentNode.parentNode.removeChild(evtTgt.parentNode)
                    }
                }

                mediaRecorder.ondataavailable = e => {
                    chunks.push(e.data)
                }
            })
            .catch(err => {
                console.log('The following error occurred: ' + err)
            })
    }
}