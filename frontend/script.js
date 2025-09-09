// Configuration - Updated for proper backend connection
const API_BASE_URL = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
    ? 'http://127.0.0.1:8000/api' 
    : 'https://pingus-backend.onrender.com/api';

let currentSessionId = null;
let conversationHistory = [];
let requestTimeout = 45000; // 45 seconds timeout
let retryCount = 0;
const maxRetries = 3;

// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Log the API URL being used
    console.log('Using API Base URL:', API_BASE_URL);
    
    // Initialize existing functionality
    initializeMobileMenu();
    initializeHeaderScroll();
    initializeAnimations();
    initializeChatbot();
    initializeDocumentManagement();
    initializeFormHandlers();
    
    // Load initial data
    loadDocumentList();
    checkSystemHealth();
}

// Enhanced Chatbot functionality with proper error handling
function initializeChatbot() {
    const chatbotTrigger = document.getElementById('chatbotTrigger');
    const chatPopup = document.getElementById('chatPopup');
    const chatInput = document.getElementById('chatInput');
    const sendButton = document.getElementById('sendChatBtn');
    const closeButton = document.getElementById('closeChatBtn');
    const messagesContainer = document.getElementById('chatMessages');
    
    console.log('Initializing enhanced chatbot...');
    console.log('Chatbot elements found:', {
        trigger: !!chatbotTrigger,
        popup: !!chatPopup,
        input: !!chatInput,
        sendBtn: !!sendButton,
        closeBtn: !!closeButton,
        messages: !!messagesContainer
    });
    
    if (chatbotTrigger && chatPopup) {
        chatbotTrigger.addEventListener('click', function() {
            console.log('Chatbot trigger clicked');
            chatPopup.classList.toggle('active');
            if (chatPopup.classList.contains('active')) {
                initializeChatSession();
                // Focus on input when opened
                if (chatInput) {
                    setTimeout(() => chatInput.focus(), 300);
                }
            }
        });
    }
    
    if (closeButton) {
        closeButton.addEventListener('click', function() {
            console.log('Close button clicked');
            chatPopup.classList.remove('active');
        });
    }
    
    if (chatInput) {
        chatInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                console.log('Enter key pressed, sending message');
                sendMessage();
            }
        });
        
        // Auto-resize input as user types
        chatInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });
    }
    
    if (sendButton) {
        sendButton.addEventListener('click', function(e) {
            e.preventDefault();
            console.log('Send button clicked');
            sendMessage();
        });
    }
    
    // Initialize chat messages container with enhanced welcome message
    if (messagesContainer) {
        messagesContainer.innerHTML = `
            <div class="welcome-message">
                <p>👋 Hello! I'm your TechCraft AI assistant. I can help you with information about our services, answer questions about web development, AI solutions, and discuss how we can help bring your project to life. How can I assist you today?</p>
            </div>
        `;
    }
    
    console.log('Enhanced chatbot initialization complete');
}

function initializeChatSession() {
    if (!currentSessionId) {
        currentSessionId = generateSessionId();
        conversationHistory = [];
        console.log('New chat session created:', currentSessionId);
    }
}

function generateSessionId() {
    return 'session_' + Date.now().toString(36) + Math.random().toString(36).substr(2);
}

// Enhanced sendMessage function with retry logic and better error handling
async function sendMessage() {
    const chatInput = document.getElementById('chatInput');
    const messagesContainer = document.getElementById('chatMessages');
    const sendButton = document.getElementById('sendChatBtn');
    
    console.log('sendMessage called');
    
    if (!chatInput || !messagesContainer) {
        console.error('Required chatbot elements not found');
        return;
    }
    
    const message = chatInput.value.trim();
    if (!message) {
        console.log('Empty message, not sending');
        return;
    }
    
    console.log('Sending message:', message);
    
    // Disable input while processing
    chatInput.disabled = true;
    if (sendButton) sendButton.disabled = true;
    
    try {
        // Add user message to UI
        addMessageToUI('user', message);
        
        // Clear input and reset height
        chatInput.value = '';
        chatInput.style.height = 'auto';
        
        // Show typing indicator
        showTypingIndicator();
        
        // Attempt to send message with retry logic
        const response = await sendMessageWithRetry(message);
        
        // Hide typing indicator
        hideTypingIndicator();
        
        // Add assistant response to UI
        addMessageToUI('assistant', response.response);
        
        // Update conversation history
        conversationHistory.push({ role: 'user', content: message });
        conversationHistory.push({ role: 'assistant', content: response.response });
        
        // Update session ID if provided
        if (response.session_id) {
            currentSessionId = response.session_id;
        }
        
        // Reset retry count on success
        retryCount = 0;
        
    } catch (error) {
        console.error('Error sending message:', error);
        hideTypingIndicator();
        
        // More specific error messages
        let errorMessage = handleErrorMessage(error);
        addMessageToUI('system', errorMessage);
    } finally {
        // Re-enable input
        chatInput.disabled = false;
        if (sendButton) sendButton.disabled = false;
        chatInput.focus();
    }
}

// Enhanced message sending with retry logic
async function sendMessageWithRetry(message, attempt = 1) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), requestTimeout);
    
    try {
        const requestBody = {
            message: message,
            conversation_history: conversationHistory,
            session_id: currentSessionId
        };
        
        console.log(`Attempt ${attempt}: Making API request to:`, `${API_BASE_URL}/chat`);
        console.log('Request payload:', requestBody);
        
        const response = await fetch(`${API_BASE_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json',
            },
            body: JSON.stringify(requestBody),
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        console.log('API response status:', response.status);
        console.log('API response headers:', Object.fromEntries(response.headers.entries()));
        
        // Get response text first for better error handling
        const responseText = await response.text();
        console.log('Raw response:', responseText.substring(0, 200) + '...');
        
        if (!response.ok) {
            console.error('Response error:', responseText);
            
            // Check if it's a rate limit error and we can retry
            if ((response.status === 429 || response.status === 503) && attempt < maxRetries) {
                console.log(`Rate limit or service unavailable, retrying in ${attempt * 2} seconds...`);
                await sleep(attempt * 2000); // Wait 2, 4, 6 seconds
                return sendMessageWithRetry(message, attempt + 1);
            }
            
            throw new Error(`HTTP error! status: ${response.status} - ${responseText}`);
        }
        
        // Parse JSON response
        const data = JSON.parse(responseText);
        console.log('Parsed response success:', data.success);
        
        if (!data.success) {
            throw new Error(`API returned error: ${data.response}`);
        }
        
        return data;
        
    } catch (error) {
        clearTimeout(timeoutId);
        
        // Handle network/timeout errors with retry
        if ((error.name === 'AbortError' || error.message.includes('fetch')) && attempt < maxRetries) {
            console.log(`Network error, retrying attempt ${attempt + 1}...`);
            await sleep(attempt * 1000);
            return sendMessageWithRetry(message, attempt + 1);
        }
        
        throw error;
    }
}

// Helper function for sleep
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Enhanced error message handling
function handleErrorMessage(error) {
    let errorMessage = '⚠️ Sorry, there was an error processing your message. ';
    
    if (error.name === 'AbortError') {
        errorMessage += 'The request timed out. Please try again with a shorter message.';
    } else if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
        errorMessage += 'Network connection issue. Please check your internet connection and try again.';
    } else if (error.message.includes('404')) {
        errorMessage += 'The chat service endpoint was not found. Please contact support.';
    } else if (error.message.includes('400')) {
        errorMessage += 'Invalid request. Please try rephrasing your message.';
    } else if (error.message.includes('429')) {
        errorMessage += 'Too many requests. Please wait a moment before trying again.';
    } else if (error.message.includes('500') || error.message.includes('503')) {
        errorMessage += 'Server error. Please try again in a few moments.';
    } else if (error.message.includes('rate') || error.message.includes('limit')) {
        errorMessage += 'Rate limit reached. Please wait a moment before sending another message.';
    } else {
        errorMessage += 'Please try again or contact support if the issue persists.';
    }
    
    return errorMessage;
}

// Updated addMessageToUI function
function addMessageToUI(role, content) {
    const messagesContainer = document.getElementById('chatMessages');
    if (!messagesContainer) return;
    
    console.log('Adding message to UI:', role, content.substring(0, 50) + '...');
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    let messageHTML = `<div class="message-content">${formatMessage(content)}</div>`;
    
    // Add feedback buttons for assistant messages
    if (role === 'assistant') {
        const messageId = Date.now().toString();
        messageHTML += `
            <div class="message-actions">
                <button class="feedback-btn thumbs-up" onclick="submitFeedback(true, '${messageId}')" title="Helpful">
                    <i class="fas fa-thumbs-up"></i>
                </button>
                <button class="feedback-btn thumbs-down" onclick="submitFeedback(false, '${messageId}')" title="Not helpful">
                    <i class="fas fa-thumbs-down"></i>
                </button>
            </div>
        `;
    }
    
    messageDiv.innerHTML = messageHTML;
    messagesContainer.appendChild(messageDiv);
    
    // Smooth scroll to bottom
    messagesContainer.scrollTo({
        top: messagesContainer.scrollHeight,
        behavior: 'smooth'
    });
    
    console.log('Message added to UI successfully');
}

function formatMessage(content) {
    // Enhanced formatting for better readability
    return content
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') // Bold text
        .replace(/\*(.*?)\*/g, '<em>$1</em>') // Italic text
        .replace(/`(.*?)`/g, '<code>$1</code>'); // Inline code
}

function showTypingIndicator() {
    const messagesContainer = document.getElementById('chatMessages');
    if (!messagesContainer) return;
    
    const existingIndicator = document.getElementById('typing-indicator');
    if (existingIndicator) return; // Don't add multiple indicators
    
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message assistant typing-indicator';
    typingDiv.id = 'typing-indicator';
    typingDiv.innerHTML = `
        <div class="typing-dots">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;
    
    messagesContainer.appendChild(typingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typing-indicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// Enhanced feedback function with visual feedback
async function submitFeedback(helpful, messageId) {
    try {
        const response = await fetch(`${API_BASE_URL}/feedback`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                session_id: currentSessionId,
                message_id: messageId,
                helpful: helpful,
                feedback: null
            })
        });
        
        if (response.ok) {
            console.log('Feedback submitted successfully');
            
            // Visual feedback - highlight the selected button
            const button = event.target.closest('.feedback-btn');
            if (button) {
                // Remove active class from all buttons in this message
                const messageActions = button.closest('.message-actions');
                messageActions.querySelectorAll('.feedback-btn').forEach(btn => {
                    btn.classList.remove('active');
                });
                
                // Add active class to clicked button
                button.classList.add('active');
                
                // Show a brief thank you message
                const thankYouDiv = document.createElement('div');
                thankYouDiv.textContent = 'Thank you for your feedback!';
                thankYouDiv.style.cssText = `
                    position: absolute;
                    background: #10B981;
                    color: white;
                    padding: 0.5rem 1rem;
                    border-radius: 8px;
                    font-size: 0.8rem;
                    margin-top: 0.5rem;
                    opacity: 0;
                    transition: opacity 0.3s ease;
                `;
                
                messageActions.appendChild(thankYouDiv);
                
                // Animate in
                setTimeout(() => thankYouDiv.style.opacity = '1', 100);
                
                // Remove after 2 seconds
                setTimeout(() => {
                    thankYouDiv.style.opacity = '0';
                    setTimeout(() => thankYouDiv.remove(), 300);
                }, 2000);
            }
        }
    } catch (error) {
        console.error('Error submitting feedback:', error);
    }
}

// Updated document upload function
async function uploadDocument(file) {
    const uploadButton = document.getElementById('uploadButton');
    const uploadProgress = document.getElementById('uploadProgress');
    const fileInput = document.getElementById('documentFile');
    
    try {
        if (uploadButton) uploadButton.disabled = true;
        if (uploadProgress) uploadProgress.style.display = 'block';
        
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API_BASE_URL}/upload`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('Upload error:', errorText);
            throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
        }
        
        const result = await response.json();
        console.log('Upload result:', result);
        
        alert('Document uploaded successfully!');
        if (fileInput) fileInput.value = '';
        await loadDocumentList();
        
    } catch (error) {
        console.error('Error uploading document:', error);
        let errorMessage = 'Error uploading document: ';
        if (error.message.includes('Failed to fetch')) {
            errorMessage += 'Cannot connect to server. Please check if the backend is running.';
        } else {
            errorMessage += error.message;
        }
        alert(errorMessage);
    } finally {
        if (uploadButton) uploadButton.disabled = false;
        if (uploadProgress) uploadProgress.style.display = 'none';
    }
}

// Updated loadDocumentList function
async function loadDocumentList() {
    try {
        const response = await fetch(`${API_BASE_URL}/files`);
        
        if (!response.ok) {
            console.warn(`Documents endpoint returned ${response.status}, this might be expected if not implemented`);
            return [];
        }
        
        const documents = await response.json();
        console.log('Loaded documents:', documents);
        updateDocumentListUI(documents);
        return documents;
        
    } catch (error) {
        console.error('Error loading documents:', error);
        return [];
    }
}

// System Health Check - Updated for deployment
async function checkSystemHealth() {
    try {
        const healthUrl = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
            ? 'http://127.0.0.1:8000/health'
            : 'https://pingus-backend.onrender.com/health';
            
        console.log('Checking health at:', healthUrl);
        const response = await fetch(healthUrl);
        
        if (!response.ok) {
            throw new Error('Health check failed');
        }
        
        const health = await response.json();
        console.log('System health:', health);
        updateSystemStatus({ status: 'online', ...health });
        
    } catch (error) {
        console.error('Health check failed:', error);
        updateSystemStatus({ status: 'offline' });
    }
}

function updateSystemStatus(health) {
    const statusIndicator = document.getElementById('system-status');
    if (!statusIndicator) return;
    
    statusIndicator.className = `status ${health.status}`;
    statusIndicator.title = `System Status: ${health.status}`;
    
    // Update status text if element exists
    const statusText = document.getElementById('system-status-text');
    if (statusText) {
        statusText.textContent = health.status.charAt(0).toUpperCase() + health.status.slice(1);
    }
    
    // Add visual indicator
    statusIndicator.style.backgroundColor = health.status === 'online' ? '#4CAF50' : '#f44336';
}

// Connection test function
async function testBackendConnection() {
    console.log('Testing backend connection...');
    
    try {
        const healthUrl = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
            ? 'http://127.0.0.1:8000/health'
            : 'https://pingus-backend.onrender.com/health';
            
        const response = await fetch(healthUrl);
        
        if (response.ok) {
            const data = await response.json();
            console.log('✅ Backend connection successful:', data);
            return true;
        } else {
            console.error('❌ Backend responded with error:', response.status);
            return false;
        }
    } catch (error) {
        console.error('❌ Cannot connect to backend:', error);
        return false;
    }
}

function initializeMobileMenu() {
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    const navLinks = document.querySelector('.nav-links');
    
    if (mobileMenuBtn && navLinks) {
        mobileMenuBtn.addEventListener('click', function() {
            navLinks.classList.toggle('active');
            const icon = this.querySelector('i');
            if (navLinks.classList.contains('active')) {
                icon.classList.remove('fa-bars');
                icon.classList.add('fa-times');
            } else {
                icon.classList.remove('fa-times');
                icon.classList.add('fa-bars');
            }
        });
    }
}

function initializeHeaderScroll() {
    const header = document.querySelector('.header');
    
    if (header) {
        window.addEventListener('scroll', function() {
            if (window.scrollY > 50) {
                header.classList.add('scrolled');
            } else {
                header.classList.remove('scrolled');
            }
        });
    }
}

function initializeDocumentManagement() {
    setupFileUpload();
    setupDocumentActions();
}

function setupFileUpload() {
    const uploadForm = document.getElementById('uploadDocumentForm');
    const fileInput = document.getElementById('documentFile');
    const uploadButton = document.getElementById('uploadButton');
    const refreshButton = document.getElementById('refreshKnowledgeBase');
    
    if (uploadForm) {
        uploadForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            if (!fileInput || !fileInput.files[0]) {
                alert('Please select a file to upload');
                return;
            }
            
            const file = fileInput.files[0];
            
            // Validate file type
            const allowedTypes = ['.docx', '.pdf', '.txt'];
            const fileExt = '.' + file.name.split('.').pop().toLowerCase();
            if (!allowedTypes.includes(fileExt)) {
                alert('Only .docx, .pdf, and .txt files are supported');
                return;
            }
            
            // Validate file size (10MB max)
            if (file.size > 10 * 1024 * 1024) {
                alert('File size must be less than 10MB');
                return;
            }
            
            await uploadDocument(file);
        });
    }
    
    if (refreshButton) {
        refreshButton.addEventListener('click', refreshKnowledgeBase);
    }
}

function updateDocumentListUI(documents) {
    const container = document.getElementById('document-list');
    if (!container) return;
    
    if (documents.length === 0) {
        container.innerHTML = '<p class="no-documents">No documents in knowledge base</p>';
        return;
    }
    
    container.innerHTML = documents.map(doc => `
        <div class="document-item">
            <div class="document-info">
                <div class="document-icon">
                    <i class="fas fa-file-word"></i>
                </div>
                <div class="document-details">
                    <h4>${doc.name}</h4>
                    <p class="document-meta">Size: ${formatFileSize(doc.size)}</p>
                    <p class="document-meta">Modified: ${new Date(doc.modified * 1000).toLocaleDateString()}</p>
                    ${doc.pages ? `<p class="document-meta">Pages: ${doc.pages}</p>` : ''}
                </div>
            </div>
            <div class="document-actions">
                <button class="btn-danger delete-doc" onclick="deleteDocument('${doc.name}')" title="Delete document">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        </div>
    `).join('');
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

async function deleteDocument(filename) {
    if (!confirm(`Are you sure you want to delete ${filename}?`)) return;
    
    try {
        const response = await fetch(`${API_BASE_URL}/files/${filename}`, {
            method: 'DELETE'
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to delete document');
        }
        
        alert('Document deleted successfully');
        await loadDocumentList();
    } catch (error) {
        console.error('Error deleting document:', error);
        alert('Failed to delete document: ' + error.message);
    }
}

async function refreshKnowledgeBase() {
    const refreshButton = document.getElementById('refreshKnowledgeBase');
    
    try {
        if (refreshButton) {
            refreshButton.disabled = true;
            refreshButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Refreshing...';
        }
        
        const response = await fetch(`${API_BASE_URL}/refresh`, {
            method: 'POST'
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to refresh knowledge base');
        }
        
        alert('Knowledge base refreshed successfully');
        await loadDocumentList();
        
    } catch (error) {
        console.error('Error refreshing knowledge base:', error);
        alert('Failed to refresh knowledge base: ' + error.message);
    } finally {
        if (refreshButton) {
            refreshButton.disabled = false;
            refreshButton.innerHTML = '<i class="fas fa-sync-alt"></i> Refresh Knowledge Base';
        }
    }
}

function setupDocumentActions() {
    // Additional document management setup can be added here
}

function initializeFormHandlers() {
    const contactForm = document.getElementById('contactForm');
    
    if (contactForm) {
        contactForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const data = {
                name: formData.get('name'),
                email: formData.get('email'),
                message: formData.get('message')
            };
            
            console.log('Contact form submitted:', data);
            alert('Thank you for your message! We\'ll get back to you soon.');
            this.reset();
        });
    }
}

function initializeAnimations() {
    // Service card hover animations
    const serviceCards = document.querySelectorAll('.service-card');
    
    serviceCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-15px) scale(1.03)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });

    // Animate elements on scroll
    const animatedElements = document.querySelectorAll('.service-card, .process-step, .use-cases-list li');
    
    function checkScroll() {
        animatedElements.forEach(element => {
            const elementPosition = element.getBoundingClientRect().top;
            const screenPosition = window.innerHeight / 1.3;
            
            if (elementPosition < screenPosition) {
                element.style.opacity = 1;
                element.style.transform = 'translateY(0)';
            }
        });
    }
    
    // Initialize elements as hidden
    animatedElements.forEach(element => {
        element.style.opacity = 0;
        element.style.transform = 'translateY(50px)';
        element.style.transition = 'opacity 0.8s ease, transform 0.8s ease';
    });
    
    window.addEventListener('scroll', checkScroll);
    checkScroll();

    // Particle animation enhancement
    const particles = document.querySelectorAll('.particle');
    
    particles.forEach(particle => {
        const duration = 6 + Math.random() * 6;
        const delay = Math.random() * 5;
        
        particle.style.animationDuration = `${duration}s`;
        particle.style.animationDelay = `${delay}s`;
    });

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 100,
                    behavior: 'smooth'
                });
                
                // Close mobile menu if open
                const navLinks = document.querySelector('.nav-links');
                const mobileMenuBtn = document.getElementById('mobileMenuBtn');
                if (navLinks && navLinks.classList.contains('active')) {
                    navLinks.classList.remove('active');
                    const icon = mobileMenuBtn?.querySelector('i');
                    if (icon) {
                        icon.classList.remove('fa-times');
                        icon.classList.add('fa-bars');
                    }
                }
            }
        });
    });

    // Typing effect for hero title
    const heroTitle = document.querySelector('.hero h1');
    if (heroTitle) {
        const originalText = heroTitle.textContent;
        heroTitle.textContent = '';
        let i = 0;
        
        function typeWriter() {
            if (i < originalText.length) {
                heroTitle.textContent += originalText.charAt(i);
                i++;
                setTimeout(typeWriter, 50);
            }
        }
        
        setTimeout(typeWriter, 500);
    }
}

// Utility functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Error handling
window.addEventListener('error', function(e) {
    console.error('JavaScript error:', e.error);
});

// Test connection when the page loads
setTimeout(testBackendConnection, 1000);

// Periodic health checks
setInterval(checkSystemHealth, 60000); // Check every 60 seconds (reduced frequency)
