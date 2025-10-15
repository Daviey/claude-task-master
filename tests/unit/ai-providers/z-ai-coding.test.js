/**
 * tests/unit/ai-providers/z-ai-coding.test.js
 * Unit tests for Z.AI Coding provider
 */

import { describe, it, expect, jest, beforeEach, afterEach } from '@jest/globals';
import { ZAiCodingProvider } from '../../../src/ai-providers/z-ai-coding.js';

// Mock fetch for testing
global.fetch = jest.fn();

describe('ZAiCodingProvider', () => {
	let provider;
	let mockFetch;

	beforeEach(() => {
		provider = new ZAiCodingProvider();
		mockFetch = global.fetch;
		mockFetch.mockClear();
	});

	afterEach(() => {
		mockFetch.mockRestore();
	});

	describe('constructor', () => {
		it('should create a provider with correct name', () => {
			expect(provider.name).toBe('Z.AI Coding');
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
					message: { content: 'function test() { return "Hello"; }' }
				}],
				usage: {
					prompt_tokens: 12,
					completion_tokens: 18,
					total_tokens: 30
				}
			})
		};

		it('should generate code successfully', async () => {
			mockFetch.mockResolvedValue(mockResponse);

			const result = await provider.generateText({
				apiKey: 'test-key',
				modelId: 'glm-4.6',
				messages: [{ role: 'user', content: 'Write a hello world function' }],
				temperature: 0.3,
				maxTokens: 200
			});

			expect(mockFetch).toHaveBeenCalledWith(
				'https://api.z.ai/api/paas/v4/chat/completions',
				expect.objectContaining({
					method: 'POST',
					headers: expect.objectContaining({
						'Authorization': 'Bearer test-key'
					}),
					body: expect.stringContaining('"model":"glm-4.6"')
				})
			);

			expect(result.text).toBe('function test() { return "Hello"; }');
			expect(result.usage.inputTokens).toBe(12);
			expect(result.usage.outputTokens).toBe(18);
		});

		it('should handle API errors', async () => {
			mockFetch.mockResolvedValue({
				ok: false,
				status: 401,
				json: async () => ({ error: { message: 'Unauthorized' } })
			});

			await expect(provider.generateText({
				apiKey: 'invalid-key',
				modelId: 'glm-4.6',
				messages: [{ role: 'user', content: 'Write code' }]
			})).rejects.toThrow('Z.AI API error (401): Unauthorized');
		});
	});

	describe('generateObject', () => {
		const mockResponse = {
			ok: true,
			json: async () => ({
				choices: [{
					message: { content: '{"function": "test", "parameters": ["input"]}' }
				}],
				usage: {
					prompt_tokens: 15,
					completion_tokens: 20,
					total_tokens: 35
				}
			})
		};

		it('should generate code object successfully', async () => {
			mockFetch.mockResolvedValue(mockResponse);

			const result = await provider.generateObject({
				apiKey: 'test-key',
				modelId: 'glm-4.6',
				messages: [{ role: 'user', content: 'Create a function object' }],
				schema: { type: 'object', properties: { function: { type: 'string' } } },
				objectName: 'CodeObject'
			});

			expect(result.object).toEqual({ function: 'test', parameters: ['input'] });
			expect(result.usage.inputTokens).toBe(15);
		});

		it('should include JSON response format in request', async () => {
			mockFetch.mockResolvedValue(mockResponse);

			await provider.generateObject({
				apiKey: 'test-key',
				modelId: 'glm-4.6',
				messages: [{ role: 'user', content: 'Create structured code' }],
				schema: { type: 'object' },
				objectName: 'CodeObject'
			});

			const requestBody = JSON.parse(mockFetch.mock.calls[0][1].body);
			expect(requestBody.response_format).toEqual({ type: 'json_object' });
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
				modelId: 'glm-4.6',
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
						value: new TextEncoder().encode('data: {"choices":[{"delta":{"content":"code"}}]}\n\n')
					}),
					releaseLock: () => {}
				})
			}
		};

		it('should stream code successfully', async () => {
			mockFetch.mockResolvedValue(mockStreamResponse);

			const result = await provider.streamText({
				apiKey: 'test-key',
				modelId: 'glm-4.6',
				messages: [{ role: 'user', content: 'Stream code' }]
			});

			expect(mockFetch).toHaveBeenCalledWith(
				'https://api.z.ai/api/paas/v4/chat/completions',
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
				messages: [{ role: 'user', content: 'Write code' }]
			})).rejects.toThrow('Z.AI Coding API error during text generation: Z.AI Coding API key is required');
		});

		it('should require modelId in generateText', async () => {
			await expect(provider.generateText({
				apiKey: 'test-key',
				messages: [{ role: 'user', content: 'Write code' }]
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