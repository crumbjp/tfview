import io from 'socket.io-client';
const tf = require('@tensorflow/tfjs');
const tfvis = require('@tensorflow/tfjs-vis');

class Socket {
  constructor () {
    this.clientSocket = io('http://localhost:8080', {reconnection: false});;
    this.clientSocket.on('error', (err) => {
      console.log('Socket error', err);
    });
  }

  on(name, callback) {
    this.clientSocket.on(name, (data, responseCallback) => {
      console.log(`[socket] on ${name}`, data);
      callback(data, (resp) => {
        if(!this.clientSocket || !this.clientSocket.connected) {
          return;
        }
        console.log(`[socket] resp ${name}`, resp);
        responseCallback(resp);
      });
    });
  }

  emit(name, data, callback) {
    if(!this.clientSocket || !this.clientSocket.connected) {
      return;
    }
    console.log(`[socket] emit ${name}`, data);
    this.clientSocket.emit(name, data, (resp) => {
      console.log(`[socket] emit callback ${name}`, data);
      callback(null, resp);
    });
  }

  open() {
    if(!this.clientSocket.connected) {
      this.clientSocket.open();
    }
  }

  disconnect() {
    this.clientSocket.disconnect();
  }
}

let socket = new Socket();

window.tf = tf;
window.tfvis = tfvis;
window.socket = socket;

socket.on('disconnect', () => {
});

socket.on('model', async (req) => {
  window.model = await tf.loadLayersModel(req.modelUrl);
  window.modelUrl = req.modelUrl;
  window.weightPathPrefix = req.weightPathPrefix;
  document.querySelector(`${req.container.selector} .tf-label`).innerHTML = req.container.name;
  tfvis.show.modelSummary(document.querySelector(`${req.container.selector} .area`), window.model);
});

socket.on('scatterplot', (req) => {
  document.querySelector(`${req.container.selector} .tf-label`).innerHTML = req.container.name;
  tfvis.render.scatterplot(
    document.querySelector(`${req.container.selector} .area`),
    req.data,
    req.options
  );
});

let showFitCallbacksByPanel = {};
socket.on('showFitCallbacks', (req) => {
  document.querySelector(`${req.container.selector} .tf-label`).innerHTML = req.container.name;
  showFitCallbacksByPanel[req.container.selector] = showFitCallbacksByPanel[req.container.selector] || tfvis.show.fitCallbacks(
    document.querySelector(`${req.container.selector} .area`),
    req.metrics,
    req.options
  );
  showFitCallbacksByPanel[req.container.selector][req.field].apply(null, req.args);
});

socket.on('connect', () => {
  showFitCallbacksByPanel = {};
  for(let div of document.querySelectorAll('.panels .panel > div')) {
    div.innerHTML = '';
  }
});

setInterval(()=> {
  socket.open();
}, 1000);
