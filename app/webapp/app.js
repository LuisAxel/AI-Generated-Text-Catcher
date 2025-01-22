/**
 * Handles form submission to predict essay authenticity using the Flask API.
 *
 * @param {Event} event - The submit event triggered by the form.
 */
document.getElementById('predict-form').addEventListener('submit', function(event) {
    event.preventDefault();

    // Retrieve the file input and essay text
    const fileInput = document.getElementById('pdf-file');
    let essayText = document.getElementById('essay-text').value.trim();

    // Alert if neither file nor essay text is provided
    if (!fileInput.files.length && !essayText) {
        alert("Please upload a PDF file or enter text.");
        return;
    }

    // Add quotes around essay text if it's not already properly formatted
    if (essayText && !essayText.startsWith('"') && !essayText.endsWith('"')) {
        essayText = `"${essayText}"`; // Add quotes if missing
    }

    // Prepare request options based on whether a file or essay text is provided
    let requestOptions;
    if (fileInput.files.length > 0) {
        // Prepare FormData for PDF upload
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);
        requestOptions = {
            method: 'POST',
            body: formData
        };
    } else {
        // Prepare JSON payload for essay text
        const jsonPayload = JSON.stringify({ essay: essayText });
        requestOptions = {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: jsonPayload
        };
    }

    // Send request to Flask API
    fetch('http://127.0.0.1:8000/predict', requestOptions)
    .then(response => {
        if (!response.ok) {
            // Handle errors from API
            return response.text().then(text => {
                throw new Error(`Request failed with status ${response.status}: ${text}`);
            });
        }
        return response.json();  // Parse JSON response
    })
    .then(data => {
        // Redirect to results page with prediction data
        if (data.predictions) {
            const urlParams = new URLSearchParams();
            urlParams.set('authentic_percentage', data.authentic_percentage);
            urlParams.set('generated_percentage', data.generated_percentage);
            urlParams.set('predictions', JSON.stringify(data.predictions_rounded));
            urlParams.set('essay', data.essay);

            // Redirect to the results page with query parameters
            window.location.href = '/results.html?' + urlParams.toString();
        } else {
            alert('Error: ' + (data.error || 'Unknown error.'));
        }
    })
    .catch(error => {
        alert('An error occurred: ' + error.message);
    });
});
