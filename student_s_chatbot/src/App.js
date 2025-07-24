import React, { useState, useEffect, useRef } from 'react';
import './App.css';

// Import the chat service functions
import { initializeWebSocket, sendMessage, closeWebSocket } from './chatService';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [userId, setUserId] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const messagesEndRef = useRef(null);
  const wsRef = useRef(null); // Ref to hold the WebSocket instance

  // Function to scroll to the bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollBy({ top: messagesEndRef.current.scrollHeight, behavior: "smooth" });
  };

  // Effect to handle WebSocket connection and messages
  useEffect(() => {
    // Only try to connect if userId is set and not already connected
    if (userId && !isConnected) {
      console.log("Attempting to connect WebSocket for user:", userId);
      wsRef.current = initializeWebSocket(
        userId,
        (event) => { // onmessage handler
          const data = JSON.parse(event.data);
          setMessages((prevMessages) => [...prevMessages, { text: data.message, sender: data.type === 'notification' ? 'notification' : 'bot' }]);
          scrollToBottom();
        },
        () => { // onopen handler
          console.log('WebSocket Connected!');
          setIsConnected(true);
          // Optional: Fetch history on connect
          fetchHistory(userId);
        },
        () => { // onclose handler
          console.log('WebSocket Disconnected.');
          setIsConnected(false);
        },
        (error) => { // onerror handler
          console.error('WebSocket Error:', error);
          setIsConnected(false);
        }
      );

      // Cleanup function to close WebSocket on component unmount
      return () => {
        console.log("Cleaning up WebSocket.");
        if (wsRef.current) {
          closeWebSocket(wsRef.current);
          wsRef.current = null;
        }
      };
    }
  }, [userId, isConnected]); // Re-run when userId changes or connection status changes

  // Fetch conversation history
  const fetchHistory = async (currentUserId) => {
    try {
      const response = await fetch(`http://localhost:8000/user/${currentUserId}/history`);
      if (response.ok) {
        const history = await response.json();
        const formattedHistory = history.map(msg => ({
          text: msg.message,
          sender: msg.is_user ? 'user' : 'bot'
        }));
        setMessages(formattedHistory);
        scrollToBottom();
      } else {
        console.error('Failed to fetch history');
        // If user doesn't exist, create them
        if (response.status === 404 || response.status === 409) { // 409 Conflict if user exists but we try to create again
            await createUser(currentUserId);
        }
      }
    } catch (error) {
      console.error('Error fetching history:', error);
    }
  };

  // Create user
  const createUser = async (newUserId) => {
    try {
      const response = await fetch(`http://localhost:8000/user/${newUserId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      if (response.ok) {
        console.log('User created successfully:', newUserId);
      } else {
        const errorData = await response.json();
        console.error('Failed to create user:', errorData.detail);
      }
    } catch (error) {
      console.error('Error creating user:', error);
    }
  };


  const handleSendMessage = async () => {
    if (inputMessage.trim() === '' || !isConnected || !wsRef.current) return;

    // Add user message to state immediately
    setMessages((prevMessages) => [...prevMessages, { text: inputMessage, sender: 'user' }]);
    scrollToBottom();

    // Send message via WebSocket
    sendMessage(wsRef.current, inputMessage);

    setInputMessage('');
  };

  const handleUserIdSubmit = async () => {
    if (userId.trim() === '') return;
    // Attempt to connect WebSocket, effect hook will handle it
    if (!isConnected) {
        setIsConnected(false); // Force re-run of useEffect to try new connection
    }
    //createUser(userId); // Create user on submit (or fetch history will call it)
    fetchHistory(userId); // Fetch history, which also handles user creation if needed
  };


  return (
    <div className="App">
      <header className="App-header">
        <h1>Student Chatbot</h1>
      </header>
      <div className="chat-container">
        {!userId ? (
          <div className="user-id-input">
            <input
              type="text"
              placeholder="Enter your User ID"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              onKeyPress={(e) => { if (e.key === 'Enter') handleUserIdSubmit(); }}
            />
            <button onClick={handleUserIdSubmit}>Connect</button>
          </div>
        ) : (
          <>
            <div className="chat-messages">
              {messages.map((msg, index) => (
                <div key={index} className={`message ${msg.sender}`}>
                  {msg.text}
                </div>
              ))}
              <div ref={messagesEndRef} /> {/* Scroll target */}
            </div>
            <div className="chat-input">
              <input
                type="text"
                placeholder="Type your message..."
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={(e) => { if (e.key === 'Enter') handleSendMessage(); }}
                disabled={!isConnected}
              />
              <button onClick={handleSendMessage} disabled={!isConnected}>Send</button>
            </div>
          </>
        )}
        {!isConnected && userId && <p className="connection-status">Connecting...</p>}
        {isConnected && userId && <p className="connection-status">Connected as: {userId}</p>}
      </div>
    </div>
  );
}

export default App;