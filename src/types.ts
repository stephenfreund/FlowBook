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
    id: 'inspect',
    label: 'Inspect Cells',
    icon: 'ui-components:search',
    tooltip: 'Inspect for optimization and potential issues',
    requires_kernel: false
  },
  {
    id: 'profile',
    label: 'Profile Cells',
    icon: 'ui-components:run',
    tooltip: 'Profile code cells',
    requires_kernel: true
  },
  {
    id: 'cleanup',
    label: 'Cleanup',
    icon: 'ui-components:run',
    tooltip: 'Generate improved code with AI suggestions',
    requires_kernel: false
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

/**
 * Ferret inspection metadata structure
 */
export interface IFerretInspectMetadata {
  optimizability: number;
  readability: number;
  complexity: number;
  improvements: string[];
}

/**
 * Ferret profile metadata structure
 */
export interface IFerretProfileMetadata {
  start_time: number;
  end_time: number;
  duration: number;
  profile: string;
}

/**
 * Ferret cell metadata structure
 */
export interface IFerretMetadata {
  inspect?: IFerretInspectMetadata;
  profile?: IFerretProfileMetadata;
}
