'use strict';

const fs = require('fs');
const puppeteer = require('puppeteer');
const tf = require('@tensorflow/tfjs-node');
const TfView = require('../index.js');

const createModel = () => {
  const model = tf.sequential();
  const optimizer = tf.train.adam();
  // model.add(tf.layers.dense({inputShape: [2], units: 16, activation: 'relu', useBias: true}));
  // model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
  // model.compile({
  //   optimizer: optimizer,
  //   loss: 'meanSquaredError',
  //   metrics: ['binaryAccuracy'],
  // });
  model.add(tf.layers.dense({inputShape: [2], units: 64, activation: 'relu', useBias: true}));
  model.add(tf.layers.dense({units: 2, activation: 'softmax'}));
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });
  return model;
};

describe('index', ()=> {
  after(() => {
    return new Promise(async (resolve, reject) => {
      resolve();
    });
  });

  before(() => {
    return new Promise(async (resolve, reject) => {
      resolve();
    });
  });

  it('open tfview', () => {
    return new Promise(async (resolve, reject) => {
      try {
        let model = createModel();
        let tfView = new TfView({port: 8881, publish: 'test/tmp/'});
        await tfView.open('model-data', model);
        let browser = await puppeteer.launch({headless: false, args: ['--no-sandbox', '--disable-setuid-sandbox']});
        let page = await browser.newPage();
        await page.goto('http://127.0.0.1:8881/', {
          waitUntil: 'load'
        });
        // plot test
        tfView.emit('scatterplot', {
          container: {
            name: 'scatterplot-test',
            selector: '#panel2',
          },
          data: {
            values: [{x:0,y:0}, {x:1,y:0}, {x:0,y:1}, {x:1,y:1} ],
          },
          options: {
            xLabel: 'xlabel',
            yLabel: 'ylabel',
          }
        });
        // train xor
        const inputs = tf.tensor2d(new Float32Array([0,0,0,1,1,0,1,1]), [4,2]);
        const labels = tf.tensor2d(new Uint8Array([1,0,0,1,0,1,1,0]), [4,2]);
        await model.fit(inputs, labels, {
          batchSize: 32,
          validationData: [inputs, labels],
          epochs: 50,
          stepsPerEpoch: 1,
          shuffle: true,
          callbacks: {
            onEpochEnd: (i, e) => {
              console.log('****', i, e)
              if(Math.random() < 0.1) {
                tfView.emit('showFitCallbacks', {
                  container: {
                    name: 'Model Training',
                    selector: '#panel3',
                  },
                  metrics: ['loss', 'binaryAccuracy'],
                  field: 'onEpochEnd',
                  args: [i, e],
                });
              }
            }
          }
        });
        console.log(model.predict(inputs).arraySync())
        await model.save(`file://${tfView.options.publish}models/model-data-finished`);
        tfView.emit('trainFinished', {
          modelUrl: `/models/model-data-finished/model.json`,
        });
        await new Promise((resolve) => {
          setTimeout(() => resolve(), 1000);
        });
        let html = await page.evaluate(() => {
          return document.querySelector('body').innerHTML;
        });
        expect(html).to.have.string('<div class="tf-label">Model</div>');
        expect(html).to.have.string('<div class="tf-label">scatterplot-test</div>');
        expect(html).to.have.string('<div class="tf-label">Model Training</div>');
        expect(html).to.have.string('<div class="tf-label">Accuracy</div>');
        expect(html).to.have.string('<div class="tf-label">Confusion Matrix</div>');
        resolve();
      }catch(e) {
        reject(e);
      }
    });
  });
});
