/**
 * FlowBook JupyterLab Extension
 *
 * Provides kernel-specific plugin:
 * - flowbook:plugin - For flowbook_kernel (reproducibility enforcement)
 */

import { JupyterFrontEndPlugin } from '@jupyterlab/application';

import { flowbookPlugin } from './flowbook/plugin';

const plugins: JupyterFrontEndPlugin<void>[] = [flowbookPlugin];

export default plugins;
