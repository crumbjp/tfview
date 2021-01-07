const IMAGE_WIDTH = 28;
const IMAGE_HEIGHT = 28;
const IMAGE_CHANNELS = 1;
const IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;
const NUM_CLASSES = 10;
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

const classNames = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine'];

const doPrediction = (data) => {
  const testData = dataToTensor(data);
  const testxs = testData.xs.reshape([data.length, IMAGE_WIDTH, IMAGE_HEIGHT, 1]);
  const labels = testData.labels.argMax(-1);
  const preds = model.predict(testxs).argMax(-1);

  testxs.dispose();
  return [preds, labels];
};

const showAccuracy = async (preds, labels) => {
  const classAccuracy = await tfvis.metrics.perClassAccuracy(labels, preds);
  document.querySelector(`#panel4 .tf-label`).innerHTML = 'Accuracy';
  tfvis.show.perClassAccuracy(
    document.querySelector(`#panel4 .area`),
    classAccuracy,
    classNames);
  labels.dispose();
};

const showConfusion = async (preds, labels) => {
  const confusionMatrix = await tfvis.metrics.confusionMatrix(labels, preds);
  const container = {name: 'Confusion Matrix', tab: 'Evaluation'};
  document.querySelector(`#panel5 .tf-label`).innerHTML = 'Confusion Matrix';
  tfvis.render.confusionMatrix(
    document.querySelector(`#panel5 .area`),
    {
      values: confusionMatrix,
      tickLabels: classNames
    });
  labels.dispose();
}

socket.on('trainFinished', async (req) => {
  window.model = await tf.loadLayersModel(req.modelUrl);
  console.log('loaded');
  const [preds, labels] = doPrediction(req.data);
  showAccuracy(preds, labels);
  showConfusion(preds, labels);
});
