/**
 * FlowBook JupyterLab Extension
 *
 * Provides two kernel-specific plugins:
 * - flowbook:plugin - For flowbook_kernel (AI-powered commands)
 * - flowbook:sdc - For flowbook_sdc_kernel (SDC enforcement)
 */

import { JupyterFrontEndPlugin } from '@jupyterlab/application';

import { flowbookPlugin } from './flowbook/plugin';
import { sdcPlugin } from './sdc/plugin';

/**
 * Export both plugins as an array
 */
const plugins: JupyterFrontEndPlugin<void>[] = [
  flowbookPlugin,
  sdcPlugin
];

export default plugins;
