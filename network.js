let model;
(async function(){
    model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mnist/model.json');
    document.getElementById('result').innerText = 'Model Loaded! Draw a digit and click Predict!';
})();

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let drawing = false;

// Set canvas background to white
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);

// Mouse Events
canvas.addEventListener('mousedown', () => drawing = true);
canvas.addEventListener('mouseup', () => {
    drawing = false;
    ctx.beginPath();
});
canvas.addEventListener('mouseleave', () => {
    drawing = false;
    ctx.beginPath();
});
canvas.addEventListener('mousemove', draw);

// Touch Events
canvas.addEventListener('touchstart', (e) => { e.preventDefault(); drawing = true; });
canvas.addEventListener('touchend', () => {
    drawing = false;
    ctx.beginPath();
});
canvas.addEventListener('touchcancel', () => {
    drawing = false;
    ctx.beginPath();
});
canvas.addEventListener('touchmove', (e) => { e.preventDefault(); draw(e); });

function draw(e) {
    if (!drawing) return;
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';

    let rect = canvas.getBoundingClientRect();
    let x, y;
    if (e.touches) {
        x = e.touches[0].clientX - rect.left;
        y = e.touches[0].clientY - rect.top;
    } else {
        x = e.clientX - rect.left;
        y = e.clientY - rect.top;
    }

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

// Clear Canvas
document.getElementById('clear-button').addEventListener('click', () => {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    document.getElementById('result').innerText = 'Canvas Cleared!';
});

function preprocessCanvas(image) {
    // Resize the input image to 28x28 pixels
    let tensor = tf.browser.fromPixels(image)
        .resizeNearestNeighbor([28, 28]) // change the image size
        .mean(2) // convert to grayscale
        .toFloat()
        .invert() // invert the image (black background)
        .expandDims(0) // add batch dimension
        .expandDims(-1); // add channel dimension
    return tensor.div(255.0); // normalize between 0 and 1
}

document.getElementById('predict-button').addEventListener('click', async () => {
    let tensor = preprocessCanvas(canvas);

    // Make predictions on the preprocessed image tensor
    let predictions = await model.predict(tensor).data();
    let results = Array.from(predictions);

    // Get the index of the highest probability
    let maxIndex = results.indexOf(Math.max(...results));

    document.getElementById('result').innerText = `Prediction: ${maxIndex}`;
});

