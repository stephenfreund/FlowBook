/**
 * Cell toolbar extension for Ferret commands
 */

import { INotebookTracker, NotebookPanel } from '@jupyterlab/notebook';
import { Cell } from '@jupyterlab/cells';
// import { ToolbarButton } from '@jupyterlab/apputils';

import { FerretCommandsManager } from './manager';

/**
 * Extension that adds Ferret command buttons to individual cell toolbars
 */
export class CellToolbarExtension {
  private manager: FerretCommandsManager;
  private tracker: INotebookTracker;

  constructor(manager: FerretCommandsManager, tracker: INotebookTracker) {
    console.log("CELL TOOLBAR EXTENSION CONSTRUCTOR");
    this.manager = manager;
    this.tracker = tracker;
    this.setupCellToolbars();
  }

  /**
   * Set up event listeners to add buttons to cell toolbars
   */
  private setupCellToolbars(): void {
    this.tracker.widgetAdded.connect((sender, panel) => {
      panel.content.widgets.forEach(cell => {
        if (cell.model.type === 'code') {
          this.addCellToolbarButtons(cell, panel);
        }
      });

      panel.content.model?.cells.changed.connect(() => {
        panel.content.widgets.forEach(cell => {
          if (cell.model.type === 'code') {
            this.addCellToolbarButtons(cell, panel);
          }
        });
      });
    });
  }

  /**
   * Add Ferret command buttons to a cell's toolbar
   */
  private addCellToolbarButtons(cell: Cell, panel: NotebookPanel): void {
  
  //   console.log("CELL", cell);
  //   // Skip if button already exists
  //   if (cell.node.querySelector('.jp-FerretButton')) {
  //     console.log('Button already exists');
  //     return;
  //   }

  // // Button 1: Clear output
  // const clearButton = new ToolbarButton({
  //   className: 'clear-output-button',
  //   tooltip: 'Clear Output',
  //   icon: 'ui-components:clear',
  //   onClick: () => {
  //     (cell as any).model.outputs.clear();
  //   }
  // });

  // // Button 2: Copy cell
  // const copyButton = new ToolbarButton({
  //   className: 'copy-cell-button',
  //   tooltip: 'Copy Cell',
  //   icon: 'ui-components:copy',
  //   onClick: () => {
  //     navigator.clipboard.writeText(cell.model.sharedModel.getSource());
  //   }
  // });

  // console.log("CLEAR BUTTON", clearButton);
  // console.log("COPY BUTTON", copyButton);
  // console.log("CELL TOOLBAR", (cell as any).toolbar);

  // // Add buttons with unique identifiers
  // (cell as any).toolbar.addItem('clear-output', clearButton);
  // (cell as any).toolbar.addItem('copy-cell', copyButton);


  //   // const commands = 
    this.manager.getCommands();

    // commands.forEach(cmdInfo => {
    //   const button = new ToolbarButton({
    //     label: '',
    //     tooltip: `${cmdInfo.tooltip} (full notebook)`,
    //     icon: cmdInfo.icon,
    //     onClick: async () => {
    //       await this.manager.executeCommand(cmdInfo.id, panel);
    //     }
    // commands.forEach(cmdInfo => {
    //   // const button = new ToolbarButton({
    //   //   label: '',
    //   //   tooltip: `${cmdInfo.tooltip} (full notebook)`,
    //   //   icon: cmdInfo.icon,
    //   //   onClick: async () => {
    //   //   }
    //   // });
    //   const button = document.createElement('button');
    //   button.className = 'jp-ToolbarButton jp-FerretButton';
    //   button.title = 'Inspect this cell';
    //   button.innerHTML = '🔍';
    //   button.style.marginLeft = '5px';
    //   button.onclick = async(e) => {
    //     e.stopPropagation();
    //     await this.manager.executeCommand(cmdInfo.id, panel);
    //   };

    //   console.log("BUTTON", button);

    //   const cellToolbar = cell.node.querySelector('.jp-CellHeader');
    //   console.log("CELL TOOLBAR", cellToolbar);
    //     if(cellToolbar) {
    //       cellToolbar.appendChild(button);
    //     }
    // });
  }
}



      // const button = document.createElement('button');
      // button.className = 'jp-ToolbarButton jp-FerretButton';
      // button.title = 'Inspect this cell';
      // button.innerHTML = '🔍';
      // button.style.marginLeft = '5px';
      // button.onclick = e => {
      //   e.stopPropagation();
      //   this.inspectCurrentCell();
      // };

      // Add to cell toolbar
//     console.log("ADD CELL TOOLBAR BUTTONS");
//     const commands = this.manager.getCommands();

//     console.log("COMMANDS", commands);
//     console.log("CELL", cell);

//     const toolbar = (cell as any).toolbar;
//     if (!toolbar || toolbar.hasClass('ferret-buttons-added')) {
//       return;
//     }

//     toolbar.addClass('ferret-buttons-added');

//     commands.forEach(cmdInfo => {
//       const button = new ToolbarButton({
//         label: '',
//         tooltip: `${cmdInfo.tooltip} (full notebook)`,
//         icon: cmdInfo.icon,
//         onClick: async () => {
//           await this.manager.executeCommand(cmdInfo.id, panel);
//         }
//       });

//       toolbar.addItem(`ferret-${cmdInfo.id}-cell-button`, button);
//     });
//   }
// }
