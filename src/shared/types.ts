/**
 * Shared type definitions
 */

/**
 * Information about a Jupyter kernel
 */
export interface IKernelInfo {
  kernel_id: string;
  kernel_name: string;
  connection_file?: string;
}

/**
 * Result from executing a command
 */
export interface ICommandResult {
  notebook: any;
  metadata: any;
  total_cost: number;
  total_time: number;
}

/**
 * Request body for the /flowbook/execute API endpoint
 */
export interface IExecuteCommandRequest {
  command: string;
  notebook: any;
  kernel_id?: string;
  params?: Record<string, any>;
  selected_cell_ids?: string[];
}
