/**
 * FlowBook JupyterLab Extension
 *
 * Provides two kernel-specific plugins:
 * - flowbook:plugin - For flowbook_kernel (reproducibility enforcement)
 * - flowbook:experimental - For experimental_kernel (AI-powered commands)
 */

import { JupyterFrontEndPlugin } from '@jupyterlab/application';

import { flowbookPlugin } from './flowbook/plugin';
import { experimentalPlugin } from './experimental/plugin';

/**
 * Export both plugins as an array
 */
const plugins: JupyterFrontEndPlugin<void>[] = [
  flowbookPlugin,
  experimentalPlugin
];

export default plugins;
