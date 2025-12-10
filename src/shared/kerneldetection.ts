/**
 * Kernel detection utilities for multi-kernel extension support
 */

import { INotebookTracker, NotebookPanel } from '@jupyterlab/notebook';
import { ISignal, Signal } from '@lumino/signaling';

export type FerretKernelType = 'ferret_kernel' | 'ferret_sdc_kernel' | 'other' | 'none';

export interface IKernelChangeInfo {
  notebook: NotebookPanel;
  previousKernel: FerretKernelType;
  currentKernel: FerretKernelType;
}

/**
 * Detects and tracks kernel types for notebooks
 */
export class KernelDetector {
  private _tracker: INotebookTracker;
  private _kernelChanged = new Signal<this, IKernelChangeInfo>(this);
  private _notebookKernels = new Map<string, FerretKernelType>();

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
  getKernelType(notebook: NotebookPanel): FerretKernelType {
    const kernelName = notebook.sessionContext.session?.kernel?.name;
    return this._classifyKernel(kernelName);
  }

  /**
   * Check if notebook is using ferret_kernel
   */
  isFerretKernel(notebook: NotebookPanel): boolean {
    return this.getKernelType(notebook) === 'ferret_kernel';
  }

  /**
   * Check if notebook is using ferret_sdc_kernel
   */
  isSDCKernel(notebook: NotebookPanel): boolean {
    return this.getKernelType(notebook) === 'ferret_sdc_kernel';
  }

  private _classifyKernel(kernelName: string | undefined): FerretKernelType {
    if (!kernelName) {
      return 'none';
    }
    if (kernelName === 'ferret_kernel') {
      return 'ferret_kernel';
    }
    if (kernelName === 'ferret_sdc_kernel') {
      return 'ferret_sdc_kernel';
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
