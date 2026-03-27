const CACHE_NAME = 'nutriscan-v1';
const ASSETS = [
    '/',
    '/static/style.css',
    '/static/script.js',
    '/manifest.json',
    'https://unpkg.com/html5-qrcode',
    'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap'
];

self.addEventListener('install', event => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => cache.addAll(ASSETS))
    );
});

self.addEventListener('fetch', event => {
    // For API calls to our backend or openfoodfacts, don't use cache
    if (event.request.url.includes('/api/')) {
        return;
    }
    
    event.respondWith(
        caches.match(event.request)
            .then(response => {
                return response || fetch(event.request);
            })
    );
});
