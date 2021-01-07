const classNames = ['0', '1'];

const doPrediction = (data) => {
  const inputs = tf.tensor2d(new Float32Array([0,0,0,1,1,0,1,1]), [4,2]);
  const labels = tf.tensor2d(new Uint8Array([1,0,0,1,0,1,1,0]), [4,2]).argMax(-1);
  const preds = window.model.predict(inputs).argMax(-1);;
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
};

socket.on('trainFinished', async (req) => {
  window.model = await tf.loadLayersModel(`${location.origin}${req.modelUrl}`);
  console.log('loaded');
  const [preds, labels] = doPrediction();
  showAccuracy(preds, labels);
  showConfusion(preds, labels);
});
