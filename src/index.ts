/**
 * DataFerret JupyterLab Extension
 *
 * Provides two kernel-specific plugins:
 * - data_ferret:plugin - For ferret_kernel (AI-powered commands)
 * - data_ferret:sdc - For ferret_sdc_kernel (SDC enforcement)
 */

import { JupyterFrontEndPlugin } from '@jupyterlab/application';

import { ferretPlugin } from './ferret/plugin';
import { sdcPlugin } from './sdc/plugin';

/**
 * Export both plugins as an array
 */
const plugins: JupyterFrontEndPlugin<void>[] = [
  ferretPlugin,
  sdcPlugin
];

export default plugins;
