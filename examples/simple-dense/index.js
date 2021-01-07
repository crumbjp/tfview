'use strict';

const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
const TfView = require('tfview');
let tfView = new TfView();

const createModel = () => {
  // Create a sequential model
  const model = tf.sequential();
  // --- original ---
  // Add a single input layer
  // model.add(tf.layers.dense({inputShape: [1], units: 1, useBias: true}));
  // -----------
  model.add(tf.layers.dense({inputShape: [1], units: 2, useBias: true}));
  model.add(tf.layers.dense({inputShape: [1], units: 3, useBias: true, activation: 'relu'}));
  // -----------
  // Add an output layer
  model.add(tf.layers.dense({units: 1, useBias: true}));
  return model;
};


const convertToTensor = (data) => {
  // Wrapping these calculations in a tidy will dispose any
  // intermediate tensors.

  return tf.tidy(() => {
    // Step 1. Shuffle the data
    tf.util.shuffle(data);

    // Step 2. Convert data to Tensor
    const inputs = data.map(d => d.horsepower);
    const labels = data.map(d => d.mpg);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    //Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    };
  });
};

const trainModel = async (model, inputs, labels) => {
  // Prepare the model for training.
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });

  const batchSize = 64;
  const epochs = 100;

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: {
      onEpochEnd: (i, e) => {
        tfView.emit('showFitCallbacks', {
          container: {
            name: 'Training Performance',
            selector: '#panel3',
          },
          metrics: ['loss', 'mse'],
          options: { height: 200, callbacks: ['onEpochEnd'] },
          field: 'onEpochEnd',
          args: [i, e],
        });
      }
    }
  });
};

const testModel = (model, inputData, normalizationData) => {
  const {inputMax, inputMin, labelMin, labelMax} = normalizationData;
  // Generate predictions for a uniform range of numbers between 0 and 1;
  // We un-normalize the data by doing the inverse of the min-max scaling
  // that we did earlier.
  const [xs, preds] = tf.tidy(() => {
    const xs = tf.linspace(0, 1, 100);
    console.log(xs, xs.reshape([100, 1]))
    const preds = model.predict(xs.reshape([100, 1]));
    const unNormXs = xs
      .mul(inputMax.sub(inputMin))
      .add(inputMin);
    const unNormPreds = preds
      .mul(labelMax.sub(labelMin))
      .add(labelMin);
    // Un-normalize the data
    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  const predictedPoints = Array.from(xs).map((val, i) => {
    return {x: val, y: preds[i]};
  });

  const originalPoints = inputData.map(d => ({
    x: d.horsepower, y: d.mpg,
  }));

  tfView.emit('scatterplot', {
    container: {
      name: 'Model Predictions vs Original Data',
      selector: '#panel4',
    },
    data: {
      values: [originalPoints, predictedPoints], series: ['original', 'predicted'],
    },
    options: {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
    }
  });
};

const run = async (done) => {
  let model = createModel();
  await tfView.open('dense-test', model);
  let seed = JSON.parse(String(fs.readFileSync('seed.json')));
  let data = seed.map(car => ({
    mpg: car.Miles_per_Gallon,
    horsepower: car.Horsepower,
  })).filter(car => (car.mpg != null && car.horsepower != null));

  const values = data.map(d => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  tfView.emit('scatterplot', {
    container: {
      name: 'Horsepower v MPG',
      selector: '#panel2',
    },
    data: {
      values: values,
    },
    options: {
      xLabel: 'Horsepower',
      yLabel: 'MPG',
    }
  });

  const tensorData = convertToTensor(data);
  const {inputs, labels} = tensorData;

  // Train the model
  await trainModel(model, inputs, labels);
  console.log('Done Training');

  testModel(model, data, tensorData);
  return;
};

run().then(() => {
  // process.exit(0);
}).catch((e) => {
  console.log(e);
  process.exit(1);
});
