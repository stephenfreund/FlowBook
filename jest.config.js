/**
 * Frontend test harness config (see FRONTEND_TESTING.md).
 *
 * Built on the @jupyterlab/testing preset, which supplies the jsdom-based
 * environment and — critically — the transform/transformIgnorePatterns
 * set that makes the ESM-shipping @jupyterlab/*, @lumino/*, @jupyter/ydoc,
 * and yjs packages importable under jest.
 */
const jestJupyterLab = require('@jupyterlab/testing/lib/jest-config.js');

const baseConfig = jestJupyterLab(__dirname);

// The preset's transformIgnorePatterns whitelist omits @jupyterlab/@lumino
// themselves (inside Lab's monorepo they resolve to source). Downstream
// extensions consume them as ESM from node_modules, so they must be
// babel-transformed too — same override the official extension template uses.
const esModules = [
  '@codemirror',
  '@jupyter/', // ydoc, react-components, web-components
  '@jupyterlab/',
  '@lumino/',
  '@marijn',
  '@microsoft',
  'color',
  'exenv-es6',
  'lib0',
  'marked',
  'nanoid',
  'vscode-ws-jsonrpc',
  'y-protocols',
  'y-websocket',
  'yjs'
].join('|');

module.exports = {
  ...baseConfig,
  transformIgnorePatterns: [`/node_modules/(?!${esModules}).+`],
  testRegex: 'src/(.*/)?tests/.*\\.spec\\.ts[x]?$',
  // The built labextension bundles a copy of package.json — keep jest's
  // module map away from build outputs.
  modulePathIgnorePatterns: [
    '<rootDir>/flowbook/labextension',
    '<rootDir>/lib'
  ],
  collectCoverageFrom: [
    'src/**/*.{ts,tsx}',
    '!src/_archived/**',
    '!src/**/tests/**'
  ],
  coveragePathIgnorePatterns: ['/node_modules/', '/lib/']
};
