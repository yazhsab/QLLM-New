/**
 * JavaScript client SDK for Quantum LLM API.
 * Provides enterprise-grade client capabilities for interacting with the Quantum LLM API.
 */

class QuantumLLMClient {
  /**
   * Initialize the client.
   * 
   * @param {Object} options - Client options
   * @param {string} options.apiUrl - API URL
   * @param {string} [options.apiKey] - API key
   * @param {string} [options.username] - Username
   * @param {string} [options.password] - Password
   * @param {string} [options.token] - Authentication token
   * @param {number} [options.timeout=30000] - Request timeout in milliseconds
   */
  constructor(options = {}) {
    this.apiUrl = (options.apiUrl || 'http://localhost:8000').replace(/\/$/, '');
    this.apiKey = options.apiKey;
    this.username = options.username;
    this.password = options.password;
    this.token = options.token;
    this.timeout = options.timeout || 30000;
    
    // Authentication state
    this.authenticated = false;
    this.tokenExpiry = 0;
    this.userId = null;
    this.roles = [];
    
    // Auto-authenticate if credentials are provided
    if (this.token) {
      this._validateToken();
    } else if (this.apiKey) {
      this.authenticateWithApiKey(this.apiKey);
    } else if (this.username && this.password) {
      this.authenticate(this.username, this.password);
    }
  }
  
  /**
   * Validate the token and extract user information.
   * 
   * @private
   * @returns {boolean} True if token is valid
   */
  _validateToken() {
    try {
      // Decode token without verification to extract expiry
      const base64Url = this.token.split('.')[1];
      const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
      const jsonPayload = decodeURIComponent(atob(base64).split('').map(c => {
        return '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2);
      }).join(''));
      
      const decoded = JSON.parse(jsonPayload);
      
      // Check if token is expired
      if (decoded.exp && decoded.exp < Math.floor(Date.now() / 1000)) {
        console.warn('Token is expired');
        this.authenticated = false;
        return false;
      }
      
      // Extract user information
      this.userId = decoded.sub;
      this.roles = decoded.roles || [];
      this.tokenExpiry = decoded.exp || 0;
      
      this.authenticated = true;
      return true;
    } catch (error) {
      console.warn('Invalid token', error);
      this.authenticated = false;
      return false;
    }
  }
  
  /**
   * Get request headers with authentication.
   * 
   * @private
   * @returns {Object} Headers object
   */
  _getHeaders() {
    const headers = {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'User-Agent': 'QuantumLLMClient/1.0'
    };
    
    if (this.token) {
      headers['Authorization'] = `Bearer ${this.token}`;
    } else if (this.apiKey) {
      headers['X-API-Key'] = this.apiKey;
    }
    
    return headers;
  }
  
  /**
   * Check if client is authenticated.
   * 
   * @private
   * @throws {Error} If not authenticated or token is expired
   */
  _checkAuth() {
    if (!this.authenticated) {
      throw new Error('Not authenticated. Call authenticate() first.');
    }
    
    // Check if token is expired
    if (this.tokenExpiry > 0 && this.tokenExpiry < Math.floor(Date.now() / 1000)) {
      // Try to refresh token
      if (this.username && this.password) {
        console.info('Token expired. Refreshing...');
        this.authenticate(this.username, this.password);
      } else {
        throw new Error('Token expired. Please authenticate again.');
      }
    }
  }
  
  /**
   * Authenticate with username and password.
   * 
   * @param {string} username - Username
   * @param {string} password - Password
   * @returns {Promise<boolean>} True if authentication was successful
   */
  async authenticate(username, password) {
    try {
      const response = await fetch(`${this.apiUrl}/api/v1/token`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ username, password }),
        timeout: this.timeout
      });
      
      if (response.ok) {
        const data = await response.json();
        this.token = data.access_token;
        this.tokenExpiry = Math.floor(Date.now() / 1000) + data.expires_in;
        this.userId = data.user_id;
        this.roles = data.roles;
        this.authenticated = true;
        
        // Save credentials for auto-refresh
        this.username = username;
        this.password = password;
        
        return true;
      } else {
        const errorText = await response.text();
        console.error(`Authentication failed: ${errorText}`);
        this.authenticated = false;
        return false;
      }
    } catch (error) {
      console.error(`Authentication error: ${error.message}`);
      this.authenticated = false;
      return false;
    }
  }
  
  /**
   * Authenticate with API key.
   * 
   * @param {string} apiKey - API key
   * @returns {Promise<boolean>} True if authentication was successful
   */
  async authenticateWithApiKey(apiKey) {
    this.apiKey = apiKey;
    
    try {
      // Validate API key by making a test request
      const response = await fetch(`${this.apiUrl}/api/v1/models`, {
        method: 'GET',
        headers: {
          'X-API-Key': apiKey
        },
        timeout: this.timeout
      });
      
      if (response.ok) {
        this.authenticated = true;
        return true;
      } else {
        const errorText = await response.text();
        console.error(`API key validation failed: ${errorText}`);
        this.authenticated = false;
        return false;
      }
    } catch (error) {
      console.error(`API key validation error: ${error.message}`);
      this.authenticated = false;
      return false;
    }
  }
  
  /**
   * List available models.
   * 
   * @returns {Promise<Array>} List of models
   */
  async listModels() {
    this._checkAuth();
    
    const response = await fetch(`${this.apiUrl}/api/v1/models`, {
      method: 'GET',
      headers: this._getHeaders(),
      timeout: this.timeout
    });
    
    if (response.ok) {
      const data = await response.json();
      return data.models;
    } else {
      const errorText = await response.text();
      throw new Error(`Error listing models: ${errorText}`);
    }
  }
  
  /**
   * Generate text from the model.
   * 
   * @param {Object} options - Generation options
   * @param {string} options.prompt - Input prompt
   * @param {string} [options.modelId='qllm-base-4q'] - Model ID
   * @param {number} [options.maxLength=50] - Maximum length to generate
   * @param {number} [options.temperature=1.0] - Sampling temperature
   * @param {number} [options.topK] - Number of highest probability tokens to keep
   * @param {number} [options.topP] - Cumulative probability for nucleus sampling
   * @returns {Promise<Object>} Generated text response
   */
  async generateText(options) {
    this._checkAuth();
    
    const requestData = {
      prompt: options.prompt,
      model_id: options.modelId || 'qllm-base-4q',
      max_length: options.maxLength || 50,
      temperature: options.temperature || 1.0,
      stream: false
    };
    
    if (options.topK !== undefined) {
      requestData.top_k = options.topK;
    }
    
    if (options.topP !== undefined) {
      requestData.top_p = options.topP;
    }
    
    const response = await fetch(`${this.apiUrl}/api/v1/generate`, {
      method: 'POST',
      headers: this._getHeaders(),
      body: JSON.stringify(requestData),
      timeout: this.timeout
    });
    
    if (response.ok) {
      return await response.json();
    } else {
      const errorText = await response.text();
      throw new Error(`Error generating text: ${errorText}`);
    }
  }
  
  /**
   * Generate text from the model with streaming.
   * 
   * @param {Object} options - Generation options
   * @param {string} options.prompt - Input prompt
   * @param {string} [options.modelId='qllm-base-4q'] - Model ID
   * @param {number} [options.maxLength=50] - Maximum length to generate
   * @param {number} [options.temperature=1.0] - Sampling temperature
   * @param {number} [options.topK] - Number of highest probability tokens to keep
   * @param {number} [options.topP] - Cumulative probability for nucleus sampling
   * @param {Function} [options.onToken] - Callback for token events
   * @param {Function} [options.onComplete] - Callback for completion event
   * @param {Function} [options.onError] - Callback for error events
   * @returns {Promise<Object>} WebSocket connection
   */
  generateTextStream(options) {
    this._checkAuth();
    
    const requestData = {
      prompt: options.prompt,
      model_id: options.modelId || 'qllm-base-4q',
      max_length: options.maxLength || 50,
      temperature: options.temperature || 1.0
    };
    
    if (options.topK !== undefined) {
      requestData.top_k = options.topK;
    }
    
    if (options.topP !== undefined) {
      requestData.top_p = options.topP;
    }
    
    // Connect to WebSocket
    const wsProtocol = this.apiUrl.startsWith('https') ? 'wss' : 'ws';
    const wsUrl = `${wsProtocol}://${this.apiUrl.replace(/^https?:\/\//, '')}/api/v1/generate/stream`;
    
    const ws = new WebSocket(wsUrl);
    let generatedText = '';
    const responseData = {};
    
    ws.onopen = () => {
      // Authenticate
      ws.send(JSON.stringify({ token: this.token }));
      
      // Send request
      ws.send(JSON.stringify(requestData));
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.error) {
        if (options.onError) {
          options.onError(data.error);
        }
        ws.close();
        return;
      }
      
      if (data.event === 'start') {
        responseData.id = data.id;
        responseData.createdAt = data.created_at;
        responseData.generatedText = '';
      } else if (data.event === 'token') {
        const token = data.token;
        generatedText += token;
        
        if (options.onToken) {
          options.onToken(token, data.index);
        }
      } else if (data.event === 'complete') {
        responseData.generatedText = data.generated_text;
        responseData.processingTime = data.processing_time;
        
        if (options.onComplete) {
          options.onComplete(data.generated_text, responseData);
        }
        
        ws.close();
      }
    };
    
    ws.onerror = (error) => {
      if (options.onError) {
        options.onError(error);
      }
    };
    
    return ws;
  }
  
  /**
   * Get model information.
   * 
   * @param {string} modelId - Model ID
   * @returns {Promise<Object>} Model information
   */
  async getModelInfo(modelId) {
    const models = await this.listModels();
    
    for (const model of models) {
      if (model.id === modelId) {
        return model;
      }
    }
    
    throw new Error(`Model not found: ${modelId}`);
  }
}

// Export for Node.js and browser
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { QuantumLLMClient };
} else {
  window.QuantumLLMClient = QuantumLLMClient;
}

// Example usage in browser:
/*
const client = new QuantumLLMClient({
  apiUrl: 'http://localhost:8000',
  username: 'user',
  password: 'quantum_user_password'
});

// List models
client.listModels().then(models => {
  console.log('Available models:', models);
  
  // Generate text
  return client.generateText({
    prompt: 'Hello, quantum world!',
    modelId: 'qllm-base-4q',
    maxLength: 50,
    temperature: 0.7
  });
}).then(response => {
  console.log('Generated text:', response.generated_text);
}).catch(error => {
  console.error('Error:', error);
});

// Streaming example
const ws = client.generateTextStream({
  prompt: 'Hello, quantum world!',
  modelId: 'qllm-base-4q',
  maxLength: 50,
  temperature: 0.7,
  onToken: (token, index) => {
    console.log(`Token ${index}:`, token);
  },
  onComplete: (text, response) => {
    console.log('Complete text:', text);
    console.log('Response:', response);
  },
  onError: (error) => {
    console.error('Stream error:', error);
  }
});
*/