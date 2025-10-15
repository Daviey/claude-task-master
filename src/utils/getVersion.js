/**
 * Gets the version from build-time environment variable.
 * @returns {string} The version string or 'unknown'.
 */
export function getTaskMasterVersion() {
	return process.env.TM_PUBLIC_VERSION || 'unknown';
}
