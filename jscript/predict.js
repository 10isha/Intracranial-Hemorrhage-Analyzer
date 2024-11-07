// =================
// Define Functions
// =================

// == Simulate a click on a hidden button. == //
function simulateClick(tabID) {
    document.getElementById(tabID).click();
}

// == Starts predicting immediately when an image is submitted. == //
function predictOnLoad() {
    // Simulate a click on the predict button
    setTimeout(simulateClick.bind(null,'predict-button'), 500);
}

// Hide the seg image loading spinner initially
$('.spinner').hide();

// Predict when an image is submitted
$("#image-selector").change(function () {
    let reader = new FileReader();
    reader.onload = function () {
        let dataURL = reader.result;
        $("#selected-image").attr("src", dataURL);
        $("#displayed-image").attr("src", dataURL);
        $("#prediction-list").empty();

        // Clear previous segmentation
        var canvas = document.getElementById("myCanvas2");
        var ctx = canvas.getContext("2d");
        var img = document.getElementById("color-image");
        ctx.drawImage(img, 0, 0);

        $('.spinner').show();
    }

    let file = $("#image-selector").prop('files')[0];
    reader.readAsDataURL(file);
    setTimeout(simulateClick.bind(null, 'predict-button'), 500);
});

// Load the model
let model;
(async function () {
    model = await tf.loadModel('./model_1/model.json');
    $("#selected-image").attr("src", "./assets/ich.jpg");
    $('.progress-bar').hide();
    $('.spinner').show();
    predictOnLoad();
})();

// Function to update and display the analysis section based on severity score
function updateAnalysis(severityScore) {
    let analysisText = document.getElementById('analysis-text');
    let analysisSection = document.getElementById('analysis');

    // Update the analysis text based on the severity score
    if (severityScore >= 80) {
        analysisText.innerHTML = "A high severity level has been detected. Immediate medical evaluation and intervention are strongly advised to assess potential traumatic brain injuries.";
    } else if (severityScore >= 50) {
        analysisText.innerHTML = "Moderate severity is indicated. A thorough examination by medical professionals is recommended to ensure accurate diagnosis and care.";
    } else {
        analysisText.innerHTML = "A low severity level has been detected. While significant injury is unlikely, it is advisable to consult a healthcare provider for confirmation and peace of mind.";
    }

    // Display the analysis section now that the score is available
    analysisSection.style.display = "block";
}

// Predict when button is clicked
$("#predict-button").click(async function () {
    let image = $('#selected-image').get(0);
    let blank_image = $('#color-image').get(0);

    // Pre-process image
    let tensor = tf.fromPixels(image)
        .resizeNearestNeighbor([256, 256])
        .toFloat()
        .div(tf.scalar(255.0))
        .expandDims();

    // Make prediction
    let predictions = await model.predict(tensor).data();
    var preds = Array.from(predictions);

    // Threshold predictions and calculate segmented area
    var segmentedArea = 0;
    for (let i = 0; i < preds.length; i++) {
        preds[i] = preds[i] >= 0.7 ? 255 : 0;
        if (preds[i] === 255) segmentedArea++;
    }

    // Normalize severity score to range (0-100)
    const severityScore = (segmentedArea)/10;

    // Categorize the severity score
    let severityCategory;
    if (severityScore <= 20) {
        severityCategory = "Very Less";
    } else if (severityScore <= 40) {
        severityCategory = "Less";
    } else if (severityScore <= 60) {
        severityCategory = "Medium";
    } else if (severityScore <= 80) {
        severityCategory = "High";
    } else {
        severityCategory = "Severe";
    }

    // Display severity score and category
    let severityMessage = severityScore === 0 ? "Not severe" : `Severity Score: ${severityScore.toFixed(2)} (${severityCategory})`;
    let severityElement = document.getElementById("severity-score");
    if (severityElement) {
        severityElement.innerText = severityMessage;
    }

    // Display segmented images
    var orig_image = tf.fromPixels(image);
    var color_image = tf.fromPixels(blank_image);
    var pred_tensor = tf.tensor1d(preds, 'int32').reshape([256, 256, 1]);
    pred_tensor = pred_tensor.resizeNearestNeighbor([orig_image.shape[0], orig_image.shape[1]]);
    rgba_tensor = tf.concat([orig_image, pred_tensor], axis=-1).resizeNearestNeighbor([250, 250]);
    orig_image = orig_image.resizeNearestNeighbor([250, 250]);
    color_image = color_image.resizeNearestNeighbor([250, 250]);

    tf.toPixels(rgba_tensor, document.getElementById("myCanvas2"));
    tf.toPixels(orig_image, document.getElementById("myCanvas3"));
    tf.toPixels(color_image, document.getElementById("myCanvas4"));

    $('.spinner').hide();

    // Call updateAnalysis function to show the analysis section based on severity score
    updateAnalysis(severityScore);
});

