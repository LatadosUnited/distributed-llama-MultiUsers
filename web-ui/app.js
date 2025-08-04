document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');

    const API_URL = 'http://127.0.0.1:9990/v1/chat/completions';

    const addMessage = (text, sender) => {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message', `${sender}-message`);
        messageElement.textContent = text;
        chatBox.appendChild(messageElement);
        chatBox.scrollTop = chatBox.scrollHeight; // Scroll to bottom
    };

    const sendMessage = async () => {
        const prompt = userInput.value.trim();
        if (prompt === '') return;

        addMessage(prompt, 'user');
        userInput.value = '';

        try {
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    messages: [{ role: 'user', content: prompt }],
                    max_tokens: 256 // You can adjust this
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            const botResponse = data.generated_text || 'Sorry, I had trouble getting a response.';
            addMessage(botResponse, 'bot');

        } catch (error) {
            console.error('Error:', error);
            addMessage('Error: Could not connect to the server.', 'bot');
        }
    };

    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
});