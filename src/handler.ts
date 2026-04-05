/**
 * API Handler Helper for making requests to the flowbook server extension
 */

import { ServerConnection } from '@jupyterlab/services';

export async function requestAPI<T>(
  endPoint: string = '',
  init: RequestInit = {}
): Promise<T> {
  const settings = ServerConnection.makeSettings();
  const requestUrl = `${settings.baseUrl}flowbook/${endPoint}`;

  let response: Response;
  try {
    response = await ServerConnection.makeRequest(requestUrl, init, settings);
  } catch (error) {
    throw new ServerConnection.NetworkError(error as Error);
  }

  let data: any = await response.text();

  if (data.length > 0) {
    try {
      data = JSON.parse(data);
    } catch {
      // Response body is not JSON
    }
  }

  if (!response.ok) {
    // Extract error message from server response
    const errorMessage = data.error || data.message || data;
    throw new ServerConnection.ResponseError(response, errorMessage);
  }

  return data;
}
