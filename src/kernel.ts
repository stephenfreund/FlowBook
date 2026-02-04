/**
 * Kernel management utilities
 */

import { NotebookPanel } from '@jupyterlab/notebook';
import { showDialog, Dialog } from '@jupyterlab/apputils';
import { IKernelInfo } from './experimental/types';

/**
 * Utility functions for working with Jupyter kernels
 */
export class KernelUtils {
  /**
   * Get kernel information from a notebook panel
   */
  static getIKernelInfo(notebook: NotebookPanel): IKernelInfo | null {
    const session = notebook.sessionContext.session;

    if (!session || !session.kernel) {
      return null;
    }

    return {
      kernel_id: session.kernel.id,
      kernel_name: session.kernel.name
    };
  }

  /**
   * Ensure a kernel is running for the notebook, prompting the user to start one if needed
   */
  static async ensureKernel(
    notebook: NotebookPanel
  ): Promise<IKernelInfo | null> {
    await notebook.sessionContext.ready;

    let kernelInfo = this.getIKernelInfo(notebook);

    if (!kernelInfo) {
      const shouldStart = await showDialog({
        title: 'Kernel Required',
        body: 'This command requires a running kernel. Would you like to start one?',
        buttons: [
          Dialog.cancelButton(),
          Dialog.okButton({ label: 'Start Kernel' })
        ]
      });

      if (shouldStart.button.accept) {
        await notebook.sessionContext.startKernel();
        await notebook.sessionContext.ready;
        kernelInfo = this.getIKernelInfo(notebook);

        if (!kernelInfo) {
          showDialog({
            title: 'Error',
            body: 'Failed to start kernel',
            buttons: [Dialog.okButton()]
          });
          return null;
        }
      } else {
        return null;
      }
    }

    return kernelInfo;
  }
}
