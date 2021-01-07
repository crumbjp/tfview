'use strict';

const fs = require('fs');
const word2vec = require('word2vec.js');

const run = async () => {
  await word2vec.trainer({
    train: 'corpus.txt',
    output: 'vector.txt',
  });
  let analyzer = word2vec.analyzer('./vector.txt');
  let vec = analyzer.findVec('ダイエット');
  let cousins = analyzer.findCousin(vec, 10);
  console.log(cousins);
};

run().then(() => {
  console.log('end');
})
