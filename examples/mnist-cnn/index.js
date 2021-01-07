'use strict';

const fs = require('fs');
const tf = require('@tensorflow/tfjs-node');
const TfView = require('tfview');
let tfView = new TfView();
const mnist = require('mnist'); // this line is not needed in the browser

const IMAGE_WIDTH = 28;
const IMAGE_HEIGHT = 28;
const IMAGE_CHANNELS = 1;
const IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;
const NUM_CLASSES = 10;

const createModel = () => {
  const model = tf.sequential();


  // In the first layer of our convolutional neural network we have
  // to specify the input shape. Then we specify some parameters for
  // the convolution operation that takes place in this layer.
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));

  // The MaxPooling layer acts as a sort of downsampling using max values
  // in a region instead of averaging.
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

  // Repeat another conv2d + maxPooling stack.
  // Note that we have more filters in the convolution.
  model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

  // Now we flatten the output from the 2D filters into a 1D vector to prepare
  // it for input into our last layer. This is common practice when feeding
  // higher dimensional data to a final classification output layer.
  model.add(tf.layers.flatten());

  // Our last layer is a dense layer which has 10 output units, one for each
  // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
  const NUM_OUTPUT_CLASSES = 10;
  model.add(tf.layers.dense({
    units: NUM_OUTPUT_CLASSES,
    kernelInitializer: 'varianceScaling',
    activation: 'softmax'
  }));


  // Choose an optimizer, loss function and accuracy metric,
  // then compile and return the model
  const optimizer = tf.train.adam();
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
};

const dataToTensor = (data) => {
  const batchImagesArray = new Float32Array(data.length * IMAGE_SIZE);
  const batchLabelsArray = new Uint8Array(data.length * NUM_CLASSES);
  for (let i = 0; i < data.length; i++) {
    batchImagesArray.set(data[i].input, i * IMAGE_SIZE);
    batchLabelsArray.set(data[i].output, i * NUM_CLASSES);
  }
  const xs = tf.tensor2d(batchImagesArray, [data.length, IMAGE_SIZE]);
  const labels = tf.tensor2d(batchLabelsArray, [data.length, NUM_CLASSES]);
  return {xs, labels};
};

const train = async (model, mnistData) => {
  const [trainXs, trainYs] = tf.tidy(() => {
    const d = dataToTensor(mnistData.training);
    return [
      d.xs.reshape([mnistData.training.length, IMAGE_WIDTH, IMAGE_HEIGHT, 1]),
      d.labels
    ];
  });

  const [testXs, testYs] = tf.tidy(() => {
    const d = dataToTensor(mnistData.test);
    return [
      d.xs.reshape([mnistData.test.length, IMAGE_WIDTH, IMAGE_HEIGHT, 1]),
      d.labels
    ];
  });

  const BATCH_SIZE = 512;
  const EPOCHS = 20;
  return model.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: EPOCHS,
    shuffle: true,
    callbacks: {
      onEpochEnd: (i, e) => {
        tfView.emit('showFitCallbacks', {
          container: {
            name: 'Model Training',
            selector: '#panel2',
          },
          metrics: ['loss', 'val_loss', 'acc', 'val_acc'],
          field: 'onEpochEnd',
          args: [i, e],
        });
      },
      onBatchEnd: (i, e) => {
        if(Math.random() <  (1000 / (BATCH_SIZE * EPOCHS)) ) { // Reduce plots to avoid kill browser
          tfView.emit('showFitCallbacks', {
            container: {
              name: 'Model Training',
              selector: '#panel3',
            },
            metrics: ['loss', 'val_loss', 'acc', 'val_acc'],
            field: 'onBatchEnd',
            args: [i, e],
          });
        }
      }
    }
  });
};

const run = async (done) => {
  let model = createModel();
  await tfView.open('cnn-test', model);
  await train(model, mnist.set(8000, 2000));
  await model.save(`file://models/cnn-test-finished`);
  let predictData = mnist.set(500, 0);
  tfView.emit('trainFinished', {
    data: predictData.training,
    modelUrl: `/models/cnn-test-finished/model.json`,
  });

  return;
};

run().then(() => {
  // process.exit(0);
}).catch((e) => {
  console.log(e);
  process.exit(1);
});
