import { useState } from 'react';
import './App.css'; // You can customize App.css for styling

function App() {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = async () => {
    const prompt = userInput.trim();
    if (!prompt) return;

    const newUserMessage = { sender: 'You', text: prompt };
    setMessages((prevMessages) => [...prevMessages, newUserMessage]);
    setUserInput(''); // Clear input field
    setIsLoading(true);

    try {
      const response = await fetch('http://127.0.0.1:8000/generate', { // IMPORTANT: Match your FastAPI URL
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt: prompt }),
      });

      const data = await response.json();

      if (response.ok) {
        const newLLMMessage = { sender: 'LLM', text: data.response };
        setMessages((prevMessages) => [...prevMessages, newLLMMessage]);
      } else {
        const errorMessage = { sender: 'Error', text: data.error || 'Unknown error from LLM' };
        setMessages((prevMessages) => [...prevMessages, errorMessage]);
        console.error('API Error:', data);
      }
    } catch (error) {
      console.error('Network Error:', error);
      const networkError = { sender: 'Error', text: 'Could not connect to the LLM backend. Is it running?' };
      setMessages((prevMessages) => [...prevMessages, networkError]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { // Allow Shift+Enter for new line
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="chat-container">
      <h1>LLM Chat</h1>
      <div className="chat-output">
        {messages.map((msg, index) => (
          <div key={index} className={msg.sender === 'You' ? 'user-message' : 'llm-message'}>
            <strong>{msg.sender}:</strong> {msg.text}
          </div>
        ))}
        {isLoading && (
          <div className="llm-message loading"><strong>LLM:</strong> Thinking...</div>
        )}
      </div>
      <textarea
        value={userInput}
        onChange={(e) => setUserInput(e.target.value)}
        onKeyPress={handleKeyPress}
        placeholder="Type your message..."
        disabled={isLoading}
      />
      <button onClick={sendMessage} disabled={isLoading}>Send</button>
    </div>
  );
}

export default App;