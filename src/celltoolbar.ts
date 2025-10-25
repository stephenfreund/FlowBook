/**
 * Cell toolbar extension for Ferret commands
 */

import { INotebookTracker } from '@jupyterlab/notebook';
import { Cell } from '@jupyterlab/cells';
import { IToolbarWidgetRegistry, ToolbarButton } from '@jupyterlab/apputils';

import { FerretCommandsManager } from './manager';

/**
 * Extension that adds Ferret command buttons to individual cell toolbars
 */
export class CellToolbarExtension {
  private manager: FerretCommandsManager;
  private tracker: INotebookTracker;
  private toolbarRegistry: IToolbarWidgetRegistry;

  constructor(
    manager: FerretCommandsManager,
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
   * Register toolbar item factories for each Ferret command
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
        `ferret-${cmdInfo.id}`,
        (cell: Cell) => {
          // Create button for this cell
          const button = new ToolbarButton({
            label: '',
            tooltip: `${cmdInfo.tooltip} (full notebook)`,
            icon: cmdInfo.icon,
            enabled: true,
            onClick: async () => {
              // Get the current notebook panel
              const panel = this.tracker.currentWidget;
              if (panel) {
                console.log(
                  `Executing command ${cmdInfo.id} from cell toolbar`
                );
                await this.manager.executeCommand(cmdInfo.id, panel);
              } else {
                console.warn('No active notebook panel found');
              }
            }
          });

          return button;
        }
      );
    });

    console.log('CellToolbarExtension: Registration complete');
  }
}
