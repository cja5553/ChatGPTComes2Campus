// Load the themes dynamically
document.addEventListener('DOMContentLoaded', () => {
    fetch('themes.json')
        .then(response => response.json())
        .then(data => {
            const container = document.getElementById('themes-container');
            data.themes.forEach(theme => {
                const card = document.createElement('div');
                card.className = 'theme-card';
                card.innerText = `${theme.no}. ${theme.title}`;
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
