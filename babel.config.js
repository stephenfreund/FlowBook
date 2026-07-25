// Used only by jest (via babel-jest) to transform the ESM-shipping
// @jupyterlab/@lumino dependencies — see jest.config.js. The production
// build does not use babel.
module.exports = require('@jupyterlab/testing/lib/babel-config').default;
