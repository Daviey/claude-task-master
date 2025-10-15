/**
 * tests/unit/ai-providers/zai.test.js
 * Unit tests for Z.AI provider
 */

import { describe, it, expect, jest, beforeEach, afterEach } from '@jest/globals';
import { ZAiProvider } from '../../../src/ai-providers/zai.js';

// Mock fetch for testing
global.fetch = jest.fn();

describe('ZAiProvider', () => {
	let provider;
	let mockFetch;

	beforeEach(() => {
		provider = new ZAiProvider();
		mockFetch = global.fetch;
		mockFetch.mockClear();
	});

	afterEach(() => {
		mockFetch.mockRestore();
	});

	describe('constructor', () => {
		it('should create a provider with correct name', () => {
			expect(provider.name).toBe('Z.AI');
		});
	});

	describe('getEndpointForModel', () => {
		it('should return coding endpoint for coding models', () => {
			expect(provider.getEndpointForModel('glm-4.6-coding')).toBe('/api/coding/paas/v4');
			expect(provider.getEndpointForModel('glm-4.5-coding')).toBe('/api/coding/paas/v4');
			expect(provider.getEndpointForModel('glm-4.5-air-coding')).toBe('/api/coding/paas/v4');
		});

		it('should return API endpoint for regular models', () => {
			expect(provider.getEndpointForModel('glm-4.6')).toBe('/v1/chat/completions');
			expect(provider.getEndpointForModel('glm-4.5')).toBe('/v1/chat/completions');
			expect(provider.getEndpointForModel('glm-4.5-air')).toBe('/v1/chat/completions');
		});
	});

	describe('getRequiredApiKeyName', () => {
		it('should return Z_AI_API_KEY', () => {
			expect(provider.getRequiredApiKeyName()).toBe('Z_AI_API_KEY');
		});
	});

	describe('getClient', () => {
		it('should throw error when API key is missing', () => {
			expect(() => {
				provider.getClient({});
			}).toThrow('Z.AI API key is required.');
		});

		it('should create client with provided API key', () => {
			const client = provider.getClient({ apiKey: 'test-api-key' });
			expect(client).toBeDefined();
			expect(client.apiKey).toBe('test-api-key');
			expect(client.baseURL).toBe('https://api.z.ai');
		});

		it('should create client with custom base URL', () => {
			const client = provider.getClient({
				apiKey: 'test-api-key',
				baseURL: 'https://custom.z.ai'
			});
			expect(client.baseURL).toBe('https://custom.z.ai');
		});
	});

	describe('generateText', () => {
		const mockResponse = {
			ok: true,
			json: async () => ({
				choices: [{
					message: { content: 'Test response' }
				}],
				usage: {
					prompt_tokens: 10,
					completion_tokens: 15,
					total_tokens: 25
				}
			})
		};

		it('should generate text successfully', async () => {
			mockFetch.mockResolvedValue(mockResponse);

			const result = await provider.generateText({
				apiKey: 'test-key',
				modelId: 'glm-4.6',
				messages: [{ role: 'user', content: 'Hello' }],
				temperature: 0.7,
				maxTokens: 100
			});

			expect(mockFetch).toHaveBeenCalledWith(
				'https://api.z.ai/v1/chat/completions',
				expect.objectContaining({
					method: 'POST',
					headers: expect.objectContaining({
						'Authorization': 'Bearer test-key'
					}),
					body: expect.stringContaining('"model":"glm-4.6"')
				})
			);

			expect(result.text).toBe('Test response');
			expect(result.usage.inputTokens).toBe(10);
			expect(result.usage.outputTokens).toBe(15);
		});

		it('should use Z.AI API endpoint for regular models', async () => {
			mockFetch.mockResolvedValue(mockResponse);

			await provider.generateText({
				apiKey: 'test-key',
				modelId: 'glm-4.6',
				messages: [{ role: 'user', content: 'Write code' }]
			});

			expect(mockFetch).toHaveBeenCalledWith(
				'https://api.z.ai/v1/chat/completions',
				expect.any(Object)
			);
		});

		it('should use Z.AI Coding endpoint for coding models', async () => {
			mockFetch.mockResolvedValue(mockResponse);

			await provider.generateText({
				apiKey: 'test-key',
				modelId: 'glm-4.6-coding',
				messages: [{ role: 'user', content: 'Write code' }]
			});

			expect(mockFetch).toHaveBeenCalledWith(
				'https://api.z.ai/api/coding/paas/v4',
				expect.any(Object)
			);
		});

		it('should handle API errors', async () => {
			mockFetch.mockResolvedValue({
				ok: false,
				status: 401,
				json: async () => ({ error: { message: 'Unauthorized' } })
			});

			await expect(provider.generateText({
				apiKey: 'invalid-key',
				modelId: 'GLM-4.6',
				messages: [{ role: 'user', content: 'Hello' }]
			})).rejects.toThrow('Z.AI API error (401): Unauthorized');
		});
	});

	describe('generateObject', () => {
		const mockResponse = {
			ok: true,
			json: async () => ({
				choices: [{
					message: { content: '{"name": "test", "value": 42}' }
				}],
				usage: {
					prompt_tokens: 10,
					completion_tokens: 15,
					total_tokens: 25
				}
			})
		};

		it('should generate object successfully', async () => {
			mockFetch.mockResolvedValue(mockResponse);

			const result = await provider.generateObject({
				apiKey: 'test-key',
				modelId: 'GLM-4.6',
				messages: [{ role: 'user', content: 'Create object' }],
				schema: { type: 'object', properties: { name: { type: 'string' } } },
				objectName: 'TestObject'
			});

			expect(result.object).toEqual({ name: 'test', value: 42 });
			expect(result.usage.inputTokens).toBe(10);
		});

		it('should include JSON response format in request', async () => {
			mockFetch.mockResolvedValue(mockResponse);

			await provider.generateObject({
				apiKey: 'test-key',
				modelId: 'glm-4.6',
				messages: [{ role: 'user', content: 'Create object' }],
				schema: { type: 'object' },
				objectName: 'TestObject'
			});

			const requestBody = JSON.parse(mockFetch.mock.calls[0][1].body);
			expect(requestBody.response_format).toEqual({ type: 'json_object' });
		});

		it('should use API endpoint for regular models in object generation', async () => {
			mockFetch.mockResolvedValue(mockResponse);

			await provider.generateObject({
				apiKey: 'test-key',
				modelId: 'glm-4.5',
				messages: [{ role: 'user', content: 'Create object' }],
				schema: { type: 'object' },
				objectName: 'TestObject'
			});

			expect(mockFetch).toHaveBeenCalledWith(
				'https://api.z.ai/v1/chat/completions',
				expect.any(Object)
			);
		});

		it('should use Coding endpoint for coding models in object generation', async () => {
			mockFetch.mockResolvedValue(mockResponse);

			await provider.generateObject({
				apiKey: 'test-key',
				modelId: 'glm-4.5-coding',
				messages: [{ role: 'user', content: 'Create object' }],
				schema: { type: 'object' },
				objectName: 'TestObject'
			});

			expect(mockFetch).toHaveBeenCalledWith(
				'https://api.z.ai/api/coding/paas/v4',
				expect.any(Object)
			);
		});

		it('should handle invalid JSON response', async () => {
			mockFetch.mockResolvedValue({
				ok: true,
				json: async () => ({
					choices: [{
						message: { content: 'Invalid JSON {' }
					}]
				})
			});

			await expect(provider.generateObject({
				apiKey: 'test-key',
				modelId: 'GLM-4.6',
				messages: [{ role: 'user', content: 'Create object' }],
				schema: { type: 'object' },
				objectName: 'TestObject'
			})).rejects.toThrow('Failed to parse JSON response from Z.AI');
		});
	});

	describe('streamText', () => {
		const mockStreamResponse = {
			ok: true,
			body: {
				getReader: () => ({
					read: async () => ({
						done: true,
						value: new TextEncoder().encode('data: {"choices":[{"delta":{"content":"test"}}]}\n\n')
					}),
					releaseLock: () => {}
				})
			}
		};

		it('should use API endpoint for regular models in streaming', async () => {
			mockFetch.mockResolvedValue(mockStreamResponse);

			const result = await provider.streamText({
				apiKey: 'test-key',
				modelId: 'glm-4.6',
				messages: [{ role: 'user', content: 'Stream text' }]
			});

			expect(mockFetch).toHaveBeenCalledWith(
				'https://api.z.ai/v1/chat/completions',
				expect.objectContaining({
					method: 'POST',
					headers: expect.objectContaining({
						'Authorization': 'Bearer test-key'
					})
				})
			);
			expect(result.textStream).toBeDefined();
		});

		it('should use Coding endpoint for coding models in streaming', async () => {
			mockFetch.mockResolvedValue(mockStreamResponse);

			const result = await provider.streamText({
				apiKey: 'test-key',
				modelId: 'glm-4.6-coding',
				messages: [{ role: 'user', content: 'Stream code' }]
			});

			expect(mockFetch).toHaveBeenCalledWith(
				'https://api.z.ai/api/coding/paas/v4',
				expect.objectContaining({
					method: 'POST',
					headers: expect.objectContaining({
						'Authorization': 'Bearer test-key'
					})
				})
			);
			expect(result.textStream).toBeDefined();
		});
	});

	describe('validation', () => {
		it('should require API key in generateText', async () => {
			await expect(provider.generateText({
				modelId: 'glm-4.6',
				messages: [{ role: 'user', content: 'Hello' }]
			})).rejects.toThrow('Z.AI API key is required');
		});

		it('should require modelId in generateText', async () => {
			await expect(provider.generateText({
				apiKey: 'test-key',
				messages: [{ role: 'user', content: 'Hello' }]
			})).rejects.toThrow('Model ID is required');
		});

		it('should require messages in generateText', async () => {
			await expect(provider.generateText({
				apiKey: 'test-key',
				modelId: 'glm-4.6'
			})).rejects.toThrow('Invalid or empty messages array provided');
		});
	});
});