/**
 * API client for communicating with the Ferret server extension
 */

import { requestAPI } from './handler';
import { CommandResult, ExecuteCommandRequest } from './types';

/**
 * Ferret API client for server communication
 */
export class FerretAPI {
  /**
   * Execute a command on the server
   */
  static async executeCommand(
    request: ExecuteCommandRequest
  ): Promise<CommandResult> {
    try {
      const result = await requestAPI<CommandResult>('execute', {
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
