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
 * Response from the /ferret/list API endpoint
 */
export interface CommandListResponse {
  commands: CommandInfo[];
}

/**
 * Request body for the /ferret/execute API endpoint
 */
export interface ExecuteCommandRequest {
  command: string;
  notebook: any;
  kernel_id?: string;
  params?: Record<string, any>;
}
