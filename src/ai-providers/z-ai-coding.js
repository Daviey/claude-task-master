/**
 * src/ai-providers/z-ai-coding.js
 *
 * Z.AI Coding provider implementation for coding-focused models.
 */

import { ZaiProvider } from './z-ai-api.js';

export class ZAiCodingProvider extends ZaiProvider {
	constructor() {
		super('/api/coding/paas/v4/chat/completions', 'Z.AI Coding');
	}
}
