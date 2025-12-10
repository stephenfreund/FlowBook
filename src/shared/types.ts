/**
 * Shared type definitions for both plugins
 */

/**
 * Information about a Jupyter kernel
 */
export interface IKernelInfo {
  kernel_id: string;
  kernel_name: string;
  connection_file?: string;
}
