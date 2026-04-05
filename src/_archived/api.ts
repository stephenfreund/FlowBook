/**
 * API client for communicating with the FlowBook server extension
 */

import { requestAPI } from './handler';
import { ICommandResult, IExecuteCommandRequest } from './shared/types';

/**
 * FlowBook API client for server communication
 */
export class FlowbookAPI {
  /**
   * Execute a command on the server
   */
  static async executeCommand(
    request: IExecuteCommandRequest
  ): Promise<ICommandResult> {
    try {
      const result = await requestAPI<ICommandResult>('execute', {
        method: 'POST',
        body: JSON.stringify(request)
      });
      return result;
    } catch (error) {
      console.error('Failed to execute command:', error);
      throw error;
    }
  }

  /**
   * Get the kernel connection file path for a given kernel ID
   */
  static async getKernelConnectionFile(kernelId: string): Promise<string> {
    try {
      const result = await requestAPI<{ connection_file: string }>(
        `kernel/${kernelId}/connection`,
        {
          method: 'GET'
        }
      );
      return result.connection_file;
    } catch (error) {
      console.error('Failed to get kernel connection file:', error);
      throw error;
    }
  }
}
