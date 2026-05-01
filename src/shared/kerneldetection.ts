/**
 * Kernel detection utilities for multi-kernel extension support
 */

import { INotebookTracker, NotebookPanel } from '@jupyterlab/notebook';
import { ISignal, Signal } from '@lumino/signaling';

export type FlowbookKernelType = 'flowbook_kernel' | 'other' | 'none';

export interface IKernelChangeInfo {
  notebook: NotebookPanel;
  previousKernel: FlowbookKernelType;
  currentKernel: FlowbookKernelType;
}

/**
 * Detects and tracks kernel types for notebooks
 */
export class KernelDetector {
  private _tracker: INotebookTracker;
  private _kernelChanged = new Signal<this, IKernelChangeInfo>(this);
  private _notebookKernels = new Map<string, FlowbookKernelType>();

  constructor(tracker: INotebookTracker) {
    this._tracker = tracker;
    this._setupTracking();
  }

  get kernelChanged(): ISignal<this, IKernelChangeInfo> {
    return this._kernelChanged;
  }

  /**
   * Get the kernel type for a notebook
   */
  getKernelType(notebook: NotebookPanel): FlowbookKernelType {
    const kernelName = notebook.sessionContext.session?.kernel?.name;
    return this._classifyKernel(kernelName);
  }

  /**
   * Check if notebook is using flowbook_kernel (reproducibility)
   */
  isFlowbookKernel(notebook: NotebookPanel): boolean {
    return this.getKernelType(notebook) === 'flowbook_kernel';
  }

  private _classifyKernel(kernelName: string | undefined): FlowbookKernelType {
    if (!kernelName) {
      return 'none';
    }
    if (kernelName === 'flowbook_kernel') {
      return 'flowbook_kernel';
    }
    return 'other';
  }

  private _setupTracking(): void {
    // Track existing notebooks
    this._tracker.forEach(notebook => {
      this._monitorNotebook(notebook);
    });

    // Track new notebooks
    this._tracker.widgetAdded.connect((_, notebook) => {
      this._monitorNotebook(notebook);
    });
  }

  private _monitorNotebook(notebook: NotebookPanel): void {
    const path = notebook.context.path;

    // Initial kernel type
    const initialKernel = this.getKernelType(notebook);
    this._notebookKernels.set(path, initialKernel);

    // Listen for kernel changes
    notebook.sessionContext.kernelChanged.connect(() => {
      const previousKernel = this._notebookKernels.get(path) || 'none';
      const currentKernel = this.getKernelType(notebook);

      if (previousKernel !== currentKernel) {
        this._notebookKernels.set(path, currentKernel);
        this._kernelChanged.emit({
          notebook,
          previousKernel,
          currentKernel
        });
      }
    });

    // Cleanup on dispose
    notebook.disposed.connect(() => {
      this._notebookKernels.delete(path);
    });
  }
}
