class MoneyTalksScanner {
    constructor() {
        this.video = document.getElementById('camera-feed');
        this.canvas = document.getElementById('capture-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.resultDisplay = document.getElementById('result-display');
        this.muteToggle = document.getElementById('mute-toggle');
        
        this.captureInterval = 1000; // 1000ms interval
        this.isProcessing = false; // Prevents request pile-up
        this.isMuted = localStorage.getItem('moneytalks_muted') === 'true';
        this.useServerTTS = typeof window.speechSynthesis === 'undefined';
        
        this.init();
    }

    async init() {
        this.setupMuteToggle();
        await this.startCamera();
    }

    setupMuteToggle() {
        this.updateMuteButtonUI();
        this.muteToggle.addEventListener('click', () => {
            this.isMuted = !this.isMuted;
            localStorage.setItem('moneytalks_muted', this.isMuted);
            this.updateMuteButtonUI();
            if (this.isMuted) window.speechSynthesis.cancel();
        });
    }

    updateMuteButtonUI() {
        const stateText = this.isMuted ? 'Nyalakan Suara' : 'Bungkam Suara';
        const ariaText = this.isMuted ? 'Unmute audio' : 'Mute audio';
        this.muteToggle.textContent = stateText;
        this.muteToggle.setAttribute('aria-label', ariaText);
    }

    async startCamera() {
        try {
            // Target the rear-facing environment camera
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment' }
            });
            this.video.srcObject = stream;
            
            this.video.onloadedmetadata = () => {
                this.canvas.width = this.video.videoWidth;
                this.canvas.height = this.video.videoHeight;
                this.resultDisplay.textContent = 'Kamera aktif. Arahkan uang ke kamera.';
                this.speak('Kamera aktif. Silakan arahkan uang kertas ke kamera.');
                
                // Start the detection loop
                setInterval(() => this.captureAndDetect(), this.captureInterval);
            };
        } catch (err) {
            console.error('Camera access denied:', err);
            const errorMsg = 'Akses kamera ditolak. Silakan izinkan kamera untuk menggunakan aplikasi ini.';
            this.resultDisplay.textContent = errorMsg;
            this.speak(errorMsg);
        }
    }

    async captureAndDetect() {
        // Pause loop if the previous request hasn't finished
        if (this.isProcessing) return;
        this.isProcessing = true;

        this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        
        // Export to JPEG with 0.8 quality
        this.canvas.toBlob(async (blob) => {
            const formData = new FormData();
            formData.append('image', blob, 'frame.jpg');

            try {
                const response = await fetch('/api/detect', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) throw new Error('Network response was not ok');
                
                const data = await response.json();
                this.handleDetectionResult(data, blob);
            } catch (error) {
                console.error('Detection error:', error);
            } finally {
                this.isProcessing = false;
            }
        }, 'image/jpeg', 0.8);
    }

    handleDetectionResult(data, imageBlob) {
        if (data.detected) {
            this.updateUIAndSpeak(data.label);
            this.uploadScannedImageAsync(imageBlob, data.label);
        } else if (data.message) {
            // Handle low confidence
            this.updateUIAndSpeak(data.message);
        }
    }

    updateUIAndSpeak(text) {
        // Update aria-live region (visual & screen reader output)
        if (this.resultDisplay.textContent !== text) {
            this.resultDisplay.textContent = text;
            this.speak(text);
        }
    }

    async uploadScannedImageAsync(blob, label) {
        // Fire-and-forget asynchronous upload
        const formData = new FormData();
        formData.append('image', blob, 'scan.jpg');
        formData.append('detected_label', label);

        try {
            await fetch('/api/upload-image', {
                method: 'POST',
                body: formData
            });
        } catch (error) {
            console.error('Async upload failed:', error);
        }
    }

    speak(text) {
        if (this.isMuted) return;

        if (!this.useServerTTS) {
            // Primary: Web Speech API
            window.speechSynthesis.cancel(); // Prevent overlapping audio
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = 'id-ID';
            window.speechSynthesis.speak(utterance);
        } else {
            // Fallback: gTTS via Server
            const audioUrl = `/api/tts?label=${encodeURIComponent(text)}`;
            const audio = new Audio(audioUrl);
            audio.play().catch(e => console.error('Fallback TTS playback failed', e));
        }
    }
}

// Initialize application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new MoneyTalksScanner();
});