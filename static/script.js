document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const scannerSection = document.querySelector('.scanner-container').parentElement;
    const dataSection = document.getElementById('data-section');
    const resultSection = document.getElementById('result-section');
    const loading = document.getElementById('loading');
    const errorMessage = document.getElementById('error-message');
    const formGrid = document.getElementById('form-grid');
    
    // Inputs/Buttons
    const barcodeInput = document.getElementById('barcode-input');
    const searchBtn = document.getElementById('search-btn');
    const predictBtn = document.getElementById('predict-btn');
    const resetBtn = document.getElementById('reset-btn');
    
    let html5QrcodeScanner = null;
    let availableFeatures = [];

    // Initialize: Load features
    fetch('/api/features')
        .then(res => res.json())
        .then(data => {
            if (data.features) {
                availableFeatures = data.features;
                buildForm(data.features);
                initScanner();
            } else {
                showError("Could not load features from model.");
            }
        })
        .catch(err => showError("Server connection failed."));

    // Build Form dynamically
    function buildForm(features) {
        formGrid.innerHTML = '';
        features.forEach(f => {
            const group = document.createElement('div');
            group.className = 'form-group';
            group.innerHTML = `
                <label for="${f.id}">${f.label}</label>
                <input type="number" step="0.01" id="${f.id}" name="${f.id}" value="0">
            `;
            formGrid.appendChild(group);
        });
    }

    // Initialize Scanner
    function initScanner() {
        html5QrcodeScanner = new Html5QrcodeScanner(
            "reader",
            {
                fps: 10,
                qrbox: { width: 300, height: 150 },
                rememberLastUsedCamera: true,
                // Prioritize barcode formats
                formatsToSupport: [
                    Html5QrcodeSupportedFormats.EAN_13,
                    Html5QrcodeSupportedFormats.EAN_8,
                    Html5QrcodeSupportedFormats.UPC_A,
                    Html5QrcodeSupportedFormats.UPC_E,
                    Html5QrcodeSupportedFormats.CODE_128,
                    Html5QrcodeSupportedFormats.CODE_39,
                    Html5QrcodeSupportedFormats.QR_CODE,
                ],
                // Prefer rear camera
                videoConstraints: {
                    facingMode: "environment"
                }
            },
            /* verbose= */ false
        );
        html5QrcodeScanner.render(onScanSuccess, onScanFailure);
    }

    function onScanSuccess(decodedText, decodedResult) {
        // Stop scanning
        html5QrcodeScanner.clear();
        barcodeInput.value = decodedText;
        fetchProductData(decodedText);
    }

    function onScanFailure(error) {
        // Handle scan errors silently
    }

    // Search manually
    searchBtn.addEventListener('click', () => {
        const code = barcodeInput.value.trim();
        if (code) {
            try { if (html5QrcodeScanner) html5QrcodeScanner.clear(); } catch(e) { console.log('Scanner clear:', e); }
            fetchProductData(code);
        }
    });

    // Allow Enter key in barcode input
    barcodeInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            searchBtn.click();
        }
    });

    // Fetch OpenFoodFacts data
    function fetchProductData(barcode) {
        hideError();
        scannerSection.classList.add('hidden');
        loading.classList.remove('hidden');

        fetch(`/api/barcode/${barcode}`)
            .then(res => res.json())
            .then(data => {
                loading.classList.add('hidden');
                if (data.error) {
                    showError(data.error);
                    scannerSection.classList.remove('hidden');
                    return;
                }
                if (data.warning) {
                    showError(data.warning);
                }
                populateData(data);
            })
            .catch(err => {
                loading.classList.add('hidden');
                showError("Network error fetching barcode.");
                scannerSection.classList.remove('hidden');
            });
    }

    // Populate the form
    function populateData(data) {
        document.getElementById('product-name').textContent = data.product_name;
        
        const img = document.getElementById('product-img');
        if (data.image_url) {
            img.src = data.image_url;
            img.classList.remove('hidden');
        } else {
            img.classList.add('hidden');
        }

        // Fill inputs
        if (data.nutriments) {
            Object.keys(data.nutriments).forEach(key => {
                const input = document.getElementById(key);
                if (input) {
                    input.value = data.nutriments[key];
                }
            });
        }
        
        dataSection.classList.remove('hidden');
    }

    // Predict Action
    predictBtn.addEventListener('click', (e) => {
        e.preventDefault();
        
        // Gather values
        const valuesObj = {};
        availableFeatures.forEach(f => {
            const val = document.getElementById(f.id).value;
            valuesObj[f.id] = val ? parseFloat(val) : 0.0;
        });

        // Call backend predict
        predictBtn.disabled = true;
        predictBtn.textContent = "Predicting...";

        fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ values: valuesObj })
        })
        .then(res => res.json())
        .then(data => {
            predictBtn.disabled = false;
            predictBtn.textContent = "Predict Nutri-Score";

            if (data.error) {
                showError(data.error);
                return;
            }

            showResult(data);
        })
        .catch(err => {
            predictBtn.disabled = false;
            predictBtn.textContent = "Predict Nutri-Score";
            showError("Prediction failed due to network error.");
        });
    });

    // Show result
    function showResult(data) {
        dataSection.classList.add('hidden');
        resultSection.classList.remove('hidden');

        document.getElementById('result-emoji').textContent = data.emoji;
        
        const gradeEl = document.getElementById('result-grade');
        gradeEl.textContent = data.grade;
        gradeEl.setAttribute('data-grade', data.grade);
        
        document.getElementById('result-label').textContent = data.label;
    }

    // Reset everything
    resetBtn.addEventListener('click', () => {
        resultSection.classList.add('hidden');
        dataSection.classList.add('hidden');
        hideError();
        barcodeInput.value = '';
        scannerSection.classList.remove('hidden');
        initScanner();
    });

    function showError(msg) {
        errorMessage.classList.remove('hidden');
        document.getElementById('error-text').textContent = msg;
        setTimeout(hideError, 5000);
    }
    
    function hideError() {
        errorMessage.classList.add('hidden');
    }
});
