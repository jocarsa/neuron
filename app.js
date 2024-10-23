// Matrix and NeuralNetwork Classes

// Matrix Class Definition
class Matrix {
    constructor(rows, cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = Array(this.rows)
            .fill()
            .map(() => Array(this.cols).fill(0));
    }

    static fromArray(arr) {
        let m = new Matrix(arr.length, 1);
        for (let i = 0; i < arr.length; i++) {
            m.data[i][0] = arr[i];
        }
        return m;
    }

    toArray() {
        let arr = [];
        for (let i = 0; i < this.rows; i++) {
            arr.push(this.data[i][0]);
        }
        return arr;
    }

    randomize() {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.data[i][j] = Math.random() * 2 - 1;
            }
        }
    }

    static subtract(a, b) {
        let result = new Matrix(a.rows, a.cols);
        for (let i = 0; i < result.rows; i++) {
            for (let j = 0; j < result.cols; j++) {
                result.data[i][j] = a.data[i][j] - b.data[i][j];
            }
        }
        return result;
    }

    static transpose(matrix) {
        let result = new Matrix(matrix.cols, matrix.rows);
        for (let i = 0; i < matrix.rows; i++) {
            for (let j = 0; j < matrix.cols; j++) {
                result.data[j][i] = matrix.data[i][j];
            }
        }
        return result;
    }

    static multiply(a, b) {
        if (a.cols !== b.rows) {
            console.error('Columns of A must match rows of B.');
            return null;
        }
        let result = new Matrix(a.rows, b.cols);
        for (let i = 0; i < result.rows; i++) {
            for (let j = 0; j < result.cols; j++) {
                let sum = 0;
                for (let k = 0; k < a.cols; k++) {
                    sum += a.data[i][k] * b.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    multiply(n) {
        if (n instanceof Matrix) {
            if (this.rows !== n.rows || this.cols !== n.cols) {
                console.error('Matrices must have the same dimensions.');
                return;
            }
            for (let i = 0; i < this.rows; i++) {
                for (let j = 0; j < this.cols; j++) {
                    this.data[i][j] *= n.data[i][j];
                }
            }
        } else {
            for (let i = 0; i < this.rows; i++) {
                for (let j = 0; j < this.cols; j++) {
                    this.data[i][j] *= n;
                }
            }
        }
    }

    add(n) {
        if (n instanceof Matrix) {
            if (this.rows !== n.rows || this.cols !== n.cols) {
                console.error('Matrices must have the same dimensions.');
                return;
            }
            for (let i = 0; i < this.rows; i++) {
                for (let j = 0; j < this.cols; j++) {
                    this.data[i][j] += n.data[i][j];
                }
            }
        } else {
            for (let i = 0; i < this.rows; i++) {
                for (let j = 0; j < this.cols; j++) {
                    this.data[i][j] += n;
                }
            }
        }
    }

    map(func) {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                let val = this.data[i][j];
                this.data[i][j] = func(val, i, j);
            }
        }
    }

    static map(matrix, func) {
        let result = new Matrix(matrix.rows, matrix.cols);
        for (let i = 0; i < matrix.rows; i++) {
            for (let j = 0; j < matrix.cols; j++) {
                let val = matrix.data[i][j];
                result.data[i][j] = func(val, i, j);
            }
        }
        return result;
    }
}

// NeuralNetwork Class Definition
class NeuralNetwork {
    constructor(inputNodes, hiddenNodes, outputNodes) {
        this.inputNodes = inputNodes;
        this.hiddenNodes = hiddenNodes;
        this.outputNodes = outputNodes;

        this.weightsInputHidden = new Matrix(this.hiddenNodes, this.inputNodes);
        this.weightsInputHidden.randomize();
        this.weightsHiddenOutput = new Matrix(this.outputNodes, this.hiddenNodes);
        this.weightsHiddenOutput.randomize();

        this.biasHidden = new Matrix(this.hiddenNodes, 1);
        this.biasHidden.randomize();
        this.biasOutput = new Matrix(this.outputNodes, 1);
        this.biasOutput.randomize();

        this.learningRate = 0.1;
    }

    activate(x) {
        return 1 / (1 + Math.exp(-x));
    }

    activateDerivative(x) {
        return x * (1 - x);
    }

    predict(inputArray) {
        let inputs = Matrix.fromArray(inputArray);
        let hidden = Matrix.multiply(this.weightsInputHidden, inputs);
        hidden.add(this.biasHidden);
        hidden.map(this.activate);

        let outputs = Matrix.multiply(this.weightsHiddenOutput, hidden);
        outputs.add(this.biasOutput);
        outputs.map(this.activate);

        return outputs.toArray();
    }

    train(inputArray, targetArray) {
        let inputs = Matrix.fromArray(inputArray);
        let hidden = Matrix.multiply(this.weightsInputHidden, inputs);
        hidden.add(this.biasHidden);
        hidden.map(this.activate);

        let outputs = Matrix.multiply(this.weightsHiddenOutput, hidden);
        outputs.add(this.biasOutput);
        outputs.map(this.activate);

        let targets = Matrix.fromArray(targetArray);

        let outputErrors = Matrix.subtract(targets, outputs);

        let gradients = Matrix.map(outputs, this.activateDerivative);
        gradients.multiply(outputErrors);
        gradients.multiply(this.learningRate);

        let hiddenT = Matrix.transpose(hidden);
        let weightsHiddenOutputDeltas = Matrix.multiply(gradients, hiddenT);

        this.weightsHiddenOutput.add(weightsHiddenOutputDeltas);
        this.biasOutput.add(gradients);

        let weightsHiddenOutputT = Matrix.transpose(this.weightsHiddenOutput);
        let hiddenErrors = Matrix.multiply(weightsHiddenOutputT, outputErrors);

        let hiddenGradient = Matrix.map(hidden, this.activateDerivative);
        hiddenGradient.multiply(hiddenErrors);
        hiddenGradient.multiply(this.learningRate);

        let inputsT = Matrix.transpose(inputs);
        let weightsInputHiddenDeltas = Matrix.multiply(hiddenGradient, inputsT);

        this.weightsInputHidden.add(weightsInputHiddenDeltas);
        this.biasHidden.add(hiddenGradient);
    }
}

// Drawing Canvas Code
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
let drawing = false;
let hasTrained = false; // Track if training has started

ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);

canvas.addEventListener('mousedown', () => (drawing = true));
canvas.addEventListener('mouseup', () => {
    drawing = false;
    ctx.beginPath();
});
canvas.addEventListener('mouseleave', () => {
    drawing = false;
    ctx.beginPath();
});
canvas.addEventListener('mousemove', draw);

canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    drawing = true;
});
canvas.addEventListener('touchend', () => {
    drawing = false;
    ctx.beginPath();
});
canvas.addEventListener('touchcancel', () => {
    drawing = false;
    ctx.beginPath();
});
canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    draw(e);
});

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

document.getElementById('clear-button').addEventListener('click', () => {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    document.getElementById('result').innerText = 'Canvas Cleared!';
});

function preprocessCanvas() {
    let offScreenCanvas = document.createElement('canvas');
    offScreenCanvas.width = 28;
    offScreenCanvas.height = 28;
    let offScreenCtx = offScreenCanvas.getContext('2d');

    offScreenCtx.drawImage(canvas, 0, 0, 28, 28);

    let imageData = offScreenCtx.getImageData(0, 0, 28, 28);
    let data = imageData.data;

    let inputs = [];
    for (let i = 0; i < data.length; i += 4) {
        let r = data[i];
        let g = data[i + 1];
        let b = data[i + 2];
        let grayscale = (r + g + b) / 3;
        let inverted = 255 - grayscale;
        inputs.push(inverted / 255);
    }

    return inputs;
}

// Neural Network Parameters
const INPUT_NODES = 28 * 28;
const HIDDEN_NODES = 16;
const OUTPUT_NODES = 10;

let nn = new NeuralNetwork(INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES);

document.getElementById('train-button').addEventListener('click', () => {
    const label = parseInt(document.getElementById('label-input').value);
    if (isNaN(label) || label < 0 || label > 9) {
        document.getElementById('result').innerText = 'Please enter a valid label (0-9).';
        return;
    }

    let inputs = preprocessCanvas();
    let targets = Array(OUTPUT_NODES).fill(0);
    targets[label] = 1;

    nn.train(inputs, targets);
    hasTrained = true;  // Flag indicating training occurred

    // Re-draw the updated network visualization after training
    drawNetworkVisualization();
});

document.getElementById('predict-button').addEventListener('click', () => {
    let inputs = preprocessCanvas();
    let outputs = nn.predict(inputs);

    let maxIndex = outputs.indexOf(Math.max(...outputs));

    document.getElementById('result').innerText = `Prediction: ${maxIndex}`;

    drawNetworkVisualization();
});

// Network Visualization Code
const networkCanvas = document.getElementById('networkCanvas');
const networkCtx = networkCanvas.getContext('2d');

function drawNetworkVisualization() {
    networkCtx.clearRect(0, 0, networkCanvas.width, networkCanvas.height);
    networkCtx.beginPath(); // Start a fresh drawing context

    let inputLayer = [];
    let hiddenLayer = [];
    let outputLayer = [];

    const maxNeuronsDisplay = 20;
    const inputNeurons = Math.min(INPUT_NODES, maxNeuronsDisplay);
    const hiddenNeurons = HIDDEN_NODES;
    const outputNeurons = OUTPUT_NODES;

    const layerSpacing = networkCanvas.width / 4;

    // Calculate neuron positions
    for (let i = 0; i < inputNeurons; i++) {
        inputLayer.push({
            x: layerSpacing,
            y: ((i + 1) * networkCanvas.height) / (inputNeurons + 1),
        });
    }

    for (let i = 0; i < hiddenNeurons; i++) {
        hiddenLayer.push({
            x: layerSpacing * 2,
            y: ((i + 1) * networkCanvas.height) / (hiddenNeurons + 1),
        });
    }

    for (let i = 0; i < outputNeurons; i++) {
        outputLayer.push({
            x: layerSpacing * 3,
            y: ((i + 1) * networkCanvas.height) / (outputNeurons + 1),
        });
    }

    function drawConnections(fromLayer, toLayer, weights) {
    if (!hasTrained) return; // Only draw connections if training has started
    
    fromLayer.forEach((fromNeuron, i) => {
        toLayer.forEach((toNeuron, j) => {
            const weight = weights[j][i];  // Ensure correct weights are referenced
            const normalizedWeight = Math.min(Math.max(weight / 5, -1), 1); // Scale weight
            const opacity = Math.abs(normalizedWeight); // Adjust opacity based on weight
            
            networkCtx.beginPath();
            networkCtx.moveTo(fromNeuron.x, fromNeuron.y);
            networkCtx.lineTo(toNeuron.x, toNeuron.y);
            
            // Blue for positive weights, red for negative weights
            networkCtx.strokeStyle = weight > 0
                ? `rgba(0, 0, 255, ${opacity})`
                : `rgba(255, 0, 0, ${opacity})`;

            networkCtx.stroke();
        });
    });
}

    // Use the network’s current weights
    let weightsInputHidden = nn.weightsInputHidden.data;
    let weightsHiddenOutput = nn.weightsHiddenOutput.data;

    // Adjust weights for display based on neuron limits (in case of large networks)
    let adjustedWeightsInputHidden = weightsInputHidden.slice(0, hiddenNeurons).map((row) => row.slice(0, inputNeurons));
    let adjustedWeightsHiddenOutput = weightsHiddenOutput.slice(0, outputNeurons).map((row) => row.slice(0, hiddenNeurons));

    // Draw connections between the layers
    drawConnections(inputLayer, hiddenLayer, adjustedWeightsInputHidden);
    drawConnections(hiddenLayer, outputLayer, adjustedWeightsHiddenOutput);

    // Function to draw neurons
    function drawNeurons(layer) {
        layer.forEach((neuron) => {
            networkCtx.beginPath();
            networkCtx.arc(neuron.x, neuron.y, 5, 0, Math.PI * 2);
            networkCtx.fillStyle = '#fff'; // Neuron fill color
            networkCtx.fill();
            networkCtx.strokeStyle = '#000'; // Neuron border color
            networkCtx.stroke();
        });
    }

    // Draw neurons in each layer
    drawNeurons(inputLayer);
    drawNeurons(hiddenLayer);
    drawNeurons(outputLayer);
}

drawNetworkVisualization();

