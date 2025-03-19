document.addEventListener("DOMContentLoaded", function () {
    const chatWindow = document.getElementById("chat-window");
    const userInput = document.getElementById("user-input");
    const sendButton = document.getElementById("send-btn");

    sendButton.addEventListener("click", sendMessage);
    userInput.addEventListener("keypress", function (event) {
        if (event.key === "Enter") sendMessage();
    });

    function sendMessage() {
        const message = userInput.value.trim();
        if (message === "") return;

        // Display user message
        displayMessage(message, "user-message");
        userInput.value = "";

        // Send message to backend
        fetch(`${BACKEND_URL}/ask`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: message })
        })
        .then(response => response.json())
        .then(data => {
            displayMessage(data.answer, "bot-message");
        })
        .catch(error => {
            console.error("Error:", error);
            displayMessage("⚠️ Error connecting to server.", "bot-message");
        });
    }

    function displayMessage(text, className) {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("chat-bubble", className);
        messageDiv.textContent = text;
        chatWindow.appendChild(messageDiv);

        // Auto-scroll to bottom
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }
});
