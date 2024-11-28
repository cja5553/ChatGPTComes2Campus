// Load the themes dynamically
document.addEventListener('DOMContentLoaded', () => {
    fetch('themes.json')
        .then(response => response.json())
        .then(data => {
            const container = document.getElementById('themes-container');
            data.themes.forEach(theme => {
                const card = document.createElement('div');
                card.className = 'theme-card';
                card.innerHTML = `
                    <h3><i class="fas fa-book-open" style="color: #007BFF; margin-right: 8px;"></i>${theme.no}. ${theme.title}</h3>
                `;
                card.onclick = () => showModal(theme.title, theme.explanation);
                container.appendChild(card);
            });
        })
        .catch(error => console.error('Error loading themes:', error));
});

// Function to display the modal with details
function showModal(title, description) {
    const modal = document.createElement('div');
    modal.className = 'modal';
    modal.style.display = 'flex'; // Make the modal visible
    modal.innerHTML = `
        <div class="modal-content">
            <h2>${title}</h2>
            <p>${description}</p>
            <button onclick="closeModal(this)">Close</button>
        </div>
    `;
    document.body.appendChild(modal);
}

// Function to close the modal
function closeModal(button) {
    const modal = button.closest('.modal');
    modal.remove();
}
