/**
 * src/ai-providers/zai-cli.js
 *
 * Z.AI provider implementation using HTTP requests to Z.AI API endpoint.
 * This provider uses the Z.AI OpenAI-compatible API endpoint.
 *
 * Authentication:
 * - Uses ZAI_API_KEY environment variable
 * - Supports GLM-4.6, GLM-4.5, GLM-4.5-air models
 */

import { BaseAIProvider } from './base-provider.js';

/**
 * Provider for Z.AI integration
 *
 * Features:
 * - Supports 'GLM-4.6', 'GLM-4.5', 'GLM-4.5-air' models
 * - Uses Z.AI OpenAI-compatible API endpoint
 * - Direct HTTP integration with Z.AI API
 * - Comprehensive error handling
 */
export class ZaiProvider extends BaseAIProvider {
	constructor(endpoint = '/api/coding/paas/v4', name = 'Z.ai') {
		super();
		this.name = name;
		this.apiEndpoint = endpoint;
	}

	/**
	 * Returns the environment variable name required for this provider's API key.
	 * @returns {string} The environment variable name for the Z.AI API key
	 */
	getRequiredApiKeyName() {
		return 'ZAI_API_KEY';
	}

	/**
	 * Creates and returns a Z.AI client instance.
	 * @param {object} params - Parameters for client initialization
	 * @param {string} params.apiKey - Z.AI API key
	 * @param {string} [params.baseURL] - Optional custom API endpoint
	 * @returns {Function} Z.AI client function
	 * @throws {Error} If API key is missing or initialization fails
	 */
	getClient(params) {
		try {
			const { apiKey, baseURL } = params;

			if (!apiKey) {
				throw new Error('Z.AI API key is required.');
			}

			const endpoint = baseURL || 'https://api.z.ai';

			// Return a simple client object with the necessary data
			return {
				apiKey,
				baseURL: endpoint
			};
		} catch (error) {
			this.handleError('client initialization', error);
		}
	}

	/**
	 * Make HTTP request to Z.AI API
	 */
	async request(client, endpoint, payload) {
		const url = `${client.baseURL}${endpoint}`;

		const response = await fetch(url, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				'Authorization': `Bearer ${client.apiKey}`,
			},
			body: JSON.stringify(payload)
		});

		if (!response.ok) {
			const errorData = await response.json().catch(() => ({}));
			throw new Error(`Z.AI API error (${response.status}): ${errorData.error?.message || response.statusText}`);
		}

		return response.json();
	}

	/**
	 * Generate text using Z.AI API
	 */
	async generateText(params) {
		try {
			this.validateParams(params);
			this.validateMessages(params.messages);

			const client = this.getClient(params);
			const endpoint = this.apiEndpoint;

			const payload = {
				model: params.modelId,
				messages: params.messages,
				temperature: params.temperature,
				max_tokens: params.maxTokens || 4096,
				stream: false
			};

			const response = await this.request(client, endpoint, payload);

			if (!response.choices || response.choices.length === 0) {
				throw new Error('No response from Z.AI API');
			}

			return {
				text: response.choices[0].message.content,
				usage: {
					inputTokens: response.usage?.prompt_tokens || 0,
					outputTokens: response.usage?.completion_tokens || 0,
					totalTokens: response.usage?.total_tokens || 0
				}
			};
		} catch (error) {
			this.handleError('text generation', error);
		}
	}

	/**
	 * Generate structured object using Z.AI API (JSON mode)
	 */
	async generateObject(params) {
		try {
			this.validateParams(params);
			this.validateMessages(params.messages);

			if (!params.schema) {
				throw new Error('Schema is required for object generation');
			}
			if (!params.objectName) {
				throw new Error('Object name is required for object generation');
			}

			const client = this.getClient(params);
			const endpoint = this.apiEndpoint;

			const payload = {
				model: params.modelId,
				messages: params.messages,
				temperature: params.temperature,
				max_tokens: params.maxTokens || 4096,
				response_format: { type: 'json_object' },
				stream: false
			};

			const response = await this.request(client, endpoint, payload);

			if (!response.choices || response.choices.length === 0) {
				throw new Error('No response from Z.AI API');
			}

			const content = response.choices[0].message.content;
			let parsedObject;

			try {
				parsedObject = JSON.parse(content);
			} catch (parseError) {
				throw new Error(`Failed to parse JSON response from Z.AI: ${parseError.message}`);
			}

			return {
				object: parsedObject,
				usage: {
					inputTokens: response.usage?.prompt_tokens || 0,
					outputTokens: response.usage?.completion_tokens || 0,
					totalTokens: response.usage?.total_tokens || 0
				}
			};
		} catch (error) {
			this.handleError('object generation', error);
		}
	}

	/**
	 * Stream text using Z.AI API
	 */
	async streamText(params) {
		try {
			this.validateParams(params);
			this.validateMessages(params.messages);

			const client = this.getClient(params);
			const endpoint = this.apiEndpoint;

			const payload = {
				model: params.modelId,
				messages: params.messages,
				temperature: params.temperature,
				max_tokens: params.maxTokens || 4096,
				stream: true
			};

			const url = `${client.baseURL}${endpoint}`;

			const response = await fetch(url, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					'Authorization': `Bearer ${client.apiKey}`,
				},
				body: JSON.stringify(payload)
			});

			if (!response.ok) {
				const errorData = await response.json().catch(() => ({}));
				throw new Error(`Z.AI API error (${response.status}): ${errorData.error?.message || response.statusText}`);
			}

			if (!response.body) {
				throw new Error('No response body for streaming');
			}

			const reader = response.body.getReader();
			const decoder = new TextDecoder();
			let buffer = '';

			async function* streamGenerator() {
				try {
					while (true) {
						const { done, value } = await reader.read();
						if (done) break;

						buffer += decoder.decode(value, { stream: true });
						const lines = buffer.split('\n');
						buffer = lines.pop() || '';

						for (const line of lines) {
							const trimmedLine = line.trim();
							if (trimmedLine.startsWith('data: ') && trimmedLine !== 'data: [DONE]') {
								try {
									const data = JSON.parse(trimmedLine.slice(6));
									const delta = data.choices[0]?.delta;
									if (delta?.content) {
										yield delta.content;
									}
								} catch (parseError) {
									// Ignore malformed JSON in stream
									continue;
								}
							}
						}
					}
				} finally {
					reader.releaseLock();
				}
			}

			// Return a stream-compatible object
			return {
				textStream: streamGenerator(),
				finishReason: 'stop',
				usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 }
			};
		} catch (error) {
			this.handleError('text streaming', error);
		}
	}
}

// Specific provider instances
export class ZaiCodingProvider extends ZaiProvider {
	constructor() {
		super('/api/coding/paas/v4', 'Z.ai Coding Plan');
	}
}

export class ZaiApiProvider extends ZaiProvider {
	constructor() {
		super('/api/paas/v4', 'Z.ai API');
	}
}