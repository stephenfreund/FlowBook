/**
 * Type definitions for the Ferret JupyterLab extension
 */

/**
 * Information about a Ferret command from the server
 */
export interface CommandInfo {
  id: string;
  label: string;
  icon: string;
  tooltip: string;
  requires_kernel: boolean;
}

/**
 * Hardcoded list of available Ferret commands
 */
export const FERRET_COMMANDS: CommandInfo[] = [
  {
    id: 'analyze',
    label: 'Analyze Notebook',
    icon: 'ui-components:chart',
    tooltip: 'Analyze notebook structure and statistics',
    requires_kernel: false
  },
  {
    id: 'validate',
    label: 'Validate Notebook',
    icon: 'ui-components:check',
    tooltip: 'Validate notebook structure and check for issues',
    requires_kernel: false
  },
  {
    id: 'execute_all',
    label: 'Execute All Cells',
    icon: 'ui-components:run',
    tooltip: 'Execute all code cells and capture outputs',
    requires_kernel: true
  },
  {
    id: 'inspect_vars',
    label: 'Inspect Variables',
    icon: 'ui-components:inspect',
    tooltip: 'Inspect variables in the kernel namespace',
    requires_kernel: true
  }
];

/**
 * Result from executing a command
 */
export interface CommandResult {
  notebook: any;
  metadata: any;
}

/**
 * Information about a Jupyter kernel
 */
export interface KernelInfo {
  kernel_id: string;
  kernel_name: string;
  connection_file?: string;
}

/**
 * Request body for the /ferret/execute API endpoint
 */
export interface ExecuteCommandRequest {
  command: string;
  notebook: any;
  kernel_id?: string;
  params?: Record<string, any>;
  selected_cell_ids?: string[];
}
