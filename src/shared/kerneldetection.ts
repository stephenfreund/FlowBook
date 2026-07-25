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
  // Keyed by panel identity, NOT context.path — the path changes on
  // rename/move, and a path key captured at monitor time would orphan the
  // entry (and could cross-talk with a new notebook opened at the old path).
  private _notebookKernels = new Map<NotebookPanel, FlowbookKernelType>();

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
    // Initial kernel type
    const initialKernel = this.getKernelType(notebook);
    this._notebookKernels.set(notebook, initialKernel);

    // Listen for kernel changes
    notebook.sessionContext.kernelChanged.connect(() => {
      const previousKernel = this._notebookKernels.get(notebook) || 'none';
      const currentKernel = this.getKernelType(notebook);

      if (previousKernel !== currentKernel) {
        this._notebookKernels.set(notebook, currentKernel);
        this._kernelChanged.emit({
          notebook,
          previousKernel,
          currentKernel
        });
      }
    });

    // Cleanup on dispose
    notebook.disposed.connect(() => {
      this._notebookKernels.delete(notebook);
    });
  }
}
