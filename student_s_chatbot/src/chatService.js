// src/chatService.js
const BACKEND_WS_URL = "ws://localhost:8000/ws"; // Your WebSocket backend URL

let webSocket = null;

export const initializeWebSocket = (
  userId,
  onMessageCallback,
  onOpenCallback,
  onCloseCallback,
  onErrorCallback
) => {
  if (webSocket && webSocket.readyState === WebSocket.OPEN) {
    console.log("WebSocket already open.");
    return webSocket;
  }

  webSocket = new WebSocket(`${BACKEND_WS_URL}/${userId}`);

  webSocket.onopen = (event) => {
    console.log('WebSocket connection opened:', event);
    if (onOpenCallback) onOpenCallback();
  };

  webSocket.onmessage = (event) => {
    console.log('WebSocket message received:', event.data);
    if (onMessageCallback) onMessageCallback(event);
  };

  webSocket.onclose = (event) => {
    console.log('WebSocket connection closed:', event);
    if (onCloseCallback) onCloseCallback();
    // Attempt to reconnect if closed unexpectedly
    if (event.code !== 1000) { // 1000 is normal closure
        console.log("WebSocket closed unexpectedly. Attempting to reconnect in 3 seconds...");
        setTimeout(() => {
            // This re-initialization should ideally be handled by the component
            // or a more robust connection manager
            // For this basic example, we'll let the App.js useEffect handle it by changing isConnected state
            // When isConnected becomes false, it will try to re-initialize.
        }, 3000);
    }
  };

  webSocket.onerror = (error) => {
    console.error('WebSocket error:', error);
    if (onErrorCallback) onErrorCallback(error);
  };

  return webSocket;
};

export const sendMessage = (wsInstance, message) => {
  if (wsInstance && wsInstance.readyState === WebSocket.OPEN) {
    wsInstance.send(message);
  } else {
    console.error('WebSocket is not open. Message not sent:', message);
  }
};

export const closeWebSocket = (wsInstance) => {
  if (wsInstance && wsInstance.readyState === WebSocket.OPEN) {
    wsInstance.close(1000, "Component unmounted"); // 1000 is normal closure
  }
};