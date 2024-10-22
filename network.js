// Neural Network Configuration
const inputNeurons = 3;
const hiddenNeurons = 4;
const outputNeurons = 2;

// Neuron Positions
const neuronPositions = {
    input: [],
    hidden: [],
    output: []
};

// Canvas Setup
const canvas = document.getElementById('networkCanvas');
const ctx = canvas.getContext('2d');

// Initialize Neuron Positions
function initializeNeuronPositions() {
    const layerSpacing = canvas.width / 4;
    const maxNeurons = Math.max(inputNeurons, hiddenNeurons, outputNeurons);
    const neuronSpacing = canvas.height / (maxNeurons + 1);

    // Input Layer
    for (let i = 0; i < inputNeurons; i++) {
        neuronPositions.input.push({
            x: layerSpacing,
            y: (i + 1) * neuronSpacing + (maxNeurons - inputNeurons) * neuronSpacing / 2
        });
    }

    // Hidden Layer
    for (let i = 0; i < hiddenNeurons; i++) {
        neuronPositions.hidden.push({
            x: layerSpacing * 2,
            y: (i + 1) * neuronSpacing + (maxNeurons - hiddenNeurons) * neuronSpacing / 2
        });
    }

    // Output Layer
    for (let i = 0; i < outputNeurons; i++) {
        neuronPositions.output.push({
            x: layerSpacing * 3,
            y: (i + 1) * neuronSpacing + (maxNeurons - outputNeurons) * neuronSpacing / 2
        });
    }
}

// Draw Neurons and Connections
function drawNetwork() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw Connections Between Layers
    function drawConnections(fromLayer, toLayer) {
        fromLayer.forEach(fromNeuron => {
            toLayer.forEach(toNeuron => {
                ctx.beginPath();
                ctx.moveTo(fromNeuron.x, fromNeuron.y);
                ctx.lineTo(toNeuron.x, toNeuron.y);
                ctx.strokeStyle = '#aaa';
                ctx.stroke();
            });
        });
    }

    // Draw Connections
    drawConnections(neuronPositions.input, neuronPositions.hidden);
    drawConnections(neuronPositions.hidden, neuronPositions.output);

    // Draw Neurons
    function drawNeurons(layer) {
        layer.forEach(neuron => {
            ctx.beginPath();
            ctx.arc(neuron.x, neuron.y, 15, 0, Math.PI * 2);
            ctx.fillStyle = '#fff';
            ctx.fill();
            ctx.strokeStyle = '#000';
            ctx.stroke();
        });
    }

    drawNeurons(neuronPositions.input);
    drawNeurons(neuronPositions.hidden);
    drawNeurons(neuronPositions.output);
}

// Initialize and Draw the Network
initializeNeuronPositions();
drawNetwork();

