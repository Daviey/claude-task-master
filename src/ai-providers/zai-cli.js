/**
 * src/ai-providers/zai-cli.js
 *
 * Z.AI provider implementation using HTTP requests to Z.AI API endpoints.
 * This provider uses the Z.AI API endpoints for both general AI and coding-specific tasks.
 *
 * Authentication:
 * - Uses ZAI_API_KEY environment variable
 * - Supports both main API and Coding Plan API endpoints
 */

import { BaseAIProvider } from './base-provider.js';
import { log } from '../../scripts/modules/utils.js';

/**
 * Custom HTTP client for Z.AI API
 */
class ZaiAIClient {
	constructor(apiKey, baseURL = 'https://api.z.ai') {
		this.apiKey = apiKey;
		this.baseURL = baseURL;
	}

	/**
	 * Make HTTP request to Z.AI API
	 */
	async request(endpoint, payload) {
		const url = `${this.baseURL}${endpoint}`;

		const response = await fetch(url, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				'Authorization': `Bearer ${this.apiKey}`,
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
	async generateText(modelId, messages, options = {}) {
		const endpoint = '/api/coding/paas/v4';

		const payload = {
			model: modelId,
			messages,
			temperature: options.temperature,
			max_tokens: options.maxOutputTokens || 4096,
			stream: false
		};

		const response = await this.request(endpoint, payload);

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
	}

	/**
	 * Generate object using Z.AI API (JSON mode)
	 */
	async generateObject(modelId, messages, schema, options = {}) {
		const endpoint = '/api/coding/paas/v4';

		const payload = {
			model: modelId,
			messages,
			temperature: options.temperature,
			max_tokens: options.maxOutputTokens || 4096,
			response_format: { type: 'json_object' },
			stream: false
		};

		const response = await this.request(endpoint, payload);

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
	}

	/**
	 * Stream text using Z.AI API
	 */
	async *streamText(modelId, messages, options = {}) {
		const endpoint = '/api/coding/paas/v4';

		const payload = {
			model: modelId,
			messages,
			temperature: options.temperature,
			max_tokens: options.maxOutputTokens || 4096,
			stream: true
		};

		const url = `${this.baseURL}${endpoint}`;

		const response = await fetch(url, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				'Authorization': `Bearer ${this.apiKey}`,
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
}

/**
 * Provider for Z.AI integration
 *
 * Features:
 * - Supports 'GLM-4.6', 'GLM-4.5', 'GLM-4.5-air' models
 * - Uses Z.AI OpenAI-compatible API endpoint
 * - Direct HTTP integration with Z.AI API
 * - Comprehensive error handling
 */
export class ZaiCliProvider extends BaseAIProvider {
	constructor() {
		super();
		this.name = 'Z.AI CLI';
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
	 * @returns {ZaiAIClient} Z.AI client instance
	 * @throws {Error} If API key is missing or initialization fails
	 */
	getClient(params) {
		try {
			const { apiKey, baseURL } = params;

			if (!apiKey) {
				throw new Error('Z.AI API key is required.');
			}

			return new ZaiAIClient(apiKey, baseURL);
		} catch (error) {
			this.handleError('client initialization', error);
		}
	}

	/**
	 * Generate text using Z.AI API
	 */
	async generateText(params) {
		try {
			this.validateParams(params);
			this.validateMessages(params.messages);

			const client = this.getClient(params);
			const result = await client.generateText(
				params.modelId,
				params.messages,
				{
					temperature: params.temperature,
					maxOutputTokens: params.maxTokens
				}
			);

			return result;
		} catch (error) {
			this.handleError('text generation', error);
		}
	}

	/**
	 * Generate structured object using Z.AI API
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
			const result = await client.generateObject(
				params.modelId,
				params.messages,
				params.schema,
				{
					temperature: params.temperature,
					maxOutputTokens: params.maxTokens
				}
			);

			return result;
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
			const stream = client.streamText(
				params.modelId,
				params.messages,
				{
					temperature: params.temperature,
					maxOutputTokens: params.maxTokens
				}
			);

			// Return a stream-compatible object
			return {
				textStream: stream,
				finishReason: 'stop',
				usage: { inputTokens: 0, outputTokens: 0, totalTokens: 0 }
			};
		} catch (error) {
			this.handleError('text streaming', error);
		}
	}
}