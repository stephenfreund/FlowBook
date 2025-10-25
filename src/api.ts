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
}
