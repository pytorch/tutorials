(function() {
    if (window.starRatingInitialized) return;
    window.starRatingInitialized = true;

    let lastRating = 0;
    let debounceTimer;
    let isProcessing = false;
    const pageTitle = document.querySelector('h1')?.textContent || document.title;
    const pagePath = window.location.pathname;

    document.addEventListener('click', function(e) {
        if (!e.target.matches('.star[data-behavior="tutorial-rating"]') || isProcessing) return;

        const value = parseInt(e.target.dataset.count || e.target.dataset.value);
        const allStars = document.querySelectorAll('.star');

        allStars.forEach(s => {
            s.classList.toggle('active', parseInt(s.dataset.count || s.dataset.value) <= value);
        });

        isProcessing = true;
        clearTimeout(debounceTimer);

        // Immediately push the click event with rating data
        window.dataLayer = window.dataLayer || [];
        window.dataLayer.push({
            'event': 'star_rating',
            'Rating': value,
            'page_path': pagePath,
            'page_title': pageTitle,
            'event_name': 'click',
            'event_category': 'Page Rating'
        });

        console.log(`Sent rating for ${pageTitle}: ${value}`);
        lastRating = value;

        // Reset processing state after a short delay
        debounceTimer = setTimeout(() => {
            isProcessing = false;
        }, 500);
    });
})();
