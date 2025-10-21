/**
 * API client for communicating with the Ferret server extension
 */

import { requestAPI } from './handler';
import {
  CommandInfo,
  CommandResult,
  CommandListResponse,
  ExecuteCommandRequest
} from './types';

/**
 * Ferret API client for server communication
 */
export class FerretAPI {
  /**
   * Load all available commands from the server
   */
  static async loadCommands(): Promise<CommandInfo[]> {
    try {
      const response = await requestAPI<CommandListResponse>('list');
      console.log('Loaded ferret commands:', response.commands);
      return response.commands;
    } catch (error) {
      console.error('Failed to load ferret commands:', error);
      throw error;
    }
  }

  /**
   * Execute a command on the server
   */
  static async executeCommand(request: ExecuteCommandRequest): Promise<CommandResult> {
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
