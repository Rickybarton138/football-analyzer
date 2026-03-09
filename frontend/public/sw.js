// Service Worker for Manager Mentor PWA Share Target

const DB_NAME = 'share-target-db';
const STORE_NAME = 'shared-files';

function openDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      request.result.createObjectStore(STORE_NAME);
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

// Intercept POST to /share-target from the OS share sheet
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  if (url.pathname === '/share-target' && event.request.method === 'POST') {
    event.respondWith(
      (async () => {
        try {
          const formData = await event.request.formData();
          const file = formData.get('video');

          if (file) {
            const db = await openDB();
            const tx = db.transaction(STORE_NAME, 'readwrite');
            tx.objectStore(STORE_NAME).put(
              { file, timestamp: Date.now() },
              'shared-video'
            );
            await new Promise((resolve, reject) => {
              tx.oncomplete = () => resolve();
              tx.onerror = () => reject(tx.error);
            });
            db.close();
          }

          return Response.redirect('/upload?shared=true', 303);
        } catch (e) {
          return Response.redirect('/upload', 303);
        }
      })()
    );
  }
});

// Claim clients immediately
self.addEventListener('install', () => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});
