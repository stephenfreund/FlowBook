/**
 * Cell toolbar extension for FlowBook commands
 */

import { INotebookTracker } from '@jupyterlab/notebook';
import { Cell } from '@jupyterlab/cells';
import { IToolbarWidgetRegistry, ToolbarButton } from '@jupyterlab/apputils';

import { FlowbookCommandsManager } from './manager';

/**
 * Extension that adds FlowBook command buttons to individual cell toolbars
 */
export class CellToolbarExtension {
  private manager: FlowbookCommandsManager;
  private tracker: INotebookTracker;
  private toolbarRegistry: IToolbarWidgetRegistry;

  constructor(
    manager: FlowbookCommandsManager,
    tracker: INotebookTracker,
    toolbarRegistry: IToolbarWidgetRegistry
  ) {
    console.log('CellToolbarExtension: Initializing');
    this.manager = manager;
    this.tracker = tracker;
    this.toolbarRegistry = toolbarRegistry;
    this.registerToolbarItems();
  }

  /**
   * Register toolbar item factories for each FlowBook command
   */
  private registerToolbarItems(): void {
    const commands = this.manager.getCommands();
    console.log(
      `CellToolbarExtension: Registering ${commands.length} commands`
    );

    commands.forEach(cmdInfo => {
      // Register a factory for each command
      this.toolbarRegistry.addFactory<Cell>(
        'Cell',
        `flowbook-${cmdInfo.id}`,
        (cell: Cell) => {
          // Check if current kernel is experimental_kernel
          const panel = this.tracker.currentWidget;
          const kernelName = panel?.sessionContext.session?.kernel?.name;

          // Create button for this cell
          const button = new ToolbarButton({
            label: cmdInfo.label,
            tooltip: `${cmdInfo.tooltip} (this cell)`,
            // icon: cmdInfo.icon,
            enabled: true,
            onClick: async () => {
              // Get the current notebook panel
              const panel = this.tracker.currentWidget;
              if (panel) {
                const cellId = cell.model.id;
                console.log(
                  `Executing command ${cmdInfo.id} from cell toolbar on cell ${cellId}`
                );
                await this.manager.executeCommand(cmdInfo.id, panel, cellId);
              } else {
                console.warn('No active notebook panel found');
              }
            }
          });

          // Initially hide if not experimental_kernel
          if (kernelName !== 'experimental_kernel') {
            button.node.style.display = 'none';
          }

          // Listen for kernel changes to update visibility
          if (panel) {
            panel.sessionContext.kernelChanged.connect(() => {
              const newKernelName = panel.sessionContext.session?.kernel?.name;
              button.node.style.display =
                newKernelName === 'experimental_kernel' ? '' : 'none';
            });
          }

          return button;
        }
      );
    });

    console.log('CellToolbarExtension: Registration complete');
  }
}
