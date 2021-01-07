'use strict';

const fs = require('fs');
const http = require('http');
const express = require('express');
const compression = require('compression');
const socketio = require('socket.io');
const DEFAULT_PORT = 8080;

class TfView {
  constructor(options = {}) {
    this.options = options;
    this.options.publish = this.options.publish || '';
    this.port = options.port || DEFAULT_PORT;
    this.emitted = [];
    this.app = express();
    this.app.use(compression());
    this.server = http.Server(this.app);
    this.io = socketio(this.server, {});
    this.app.use('/models', express.static(`${this.options.publish}models`));
    this.app.use('/js', express.static(`${this.options.publish}js`));
    this.app.use('/', express.static(`${__dirname}/public`));
  }

  async open(name, model) {
    await model.save(`file://${this.options.publish}models/${name}`);
    this.io.on('connection', (socket) => {
      socket.emit('model', {
        container: { name: 'Model', selector: '#panel1'},
        modelUrl: `/models/${name}/model.json`,
      });
      for(let emitter of this.emitted) {
        emitter(socket);
      }
    });
    await this.server.listen(this.port);
  }

  emit(event, args) {
    let emitter = (socket) => {
      socket.emit(event, args);
    };
    emitter(this.io);
    this.emitted.push(emitter);
  }
};

module.exports = TfView;
