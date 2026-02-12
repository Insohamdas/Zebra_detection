import type { Zebra, Sighting, DashboardStats } from '../types';
import { realApi } from './realApi';

// Configuration: Switch between mock and real API
const USE_REAL_API = true; // Set to true to use real backend

// Use let instead of const so we can modify the arrays
let MOCK_ZEBRAS: Zebra[] = Array.from({ length: 20 }).map((_, i) => ({
  id: `ZEB-${1000 + i}`,
  name: `Zebra ${1000 + i}`,
  status: i < 3 ? 'new' : 'known',
  firstSeen: new Date(Date.now() - Math.random() * 10000000000).toISOString(),
  lastSeen: new Date(Date.now() - Math.random() * 100000000).toISOString(),
  sightingsCount: Math.floor(Math.random() * 50) + 1,
  thumbnailUrl: `https://picsum.photos/seed/${i}/200/200`,
  notes: i % 5 === 0 ? 'Has a distinctive scar on left flank.' : undefined,
  tags: i % 3 === 0 ? ['North Zone'] : ['South Zone'],
}));

let MOCK_SIGHTINGS: Sighting[] = Array.from({ length: 50 }).map((_, i) => ({
  id: `SIGHT-${5000 + i}`,
  zebraId: MOCK_ZEBRAS[Math.floor(Math.random() * MOCK_ZEBRAS.length)].id,
  timestamp: new Date(Date.now() - Math.random() * 604800000).toISOString(),
  location: ['Waterhole A', 'Savanna North', 'River Crossing', 'Camp Cam 1'][Math.floor(Math.random() * 4)],
  imageUrl: `https://picsum.photos/seed/sight${i}/400/300`,
  confidence: 0.85 + Math.random() * 0.14,
}));

// Check if backend is available
let backendAvailable = false;
if (USE_REAL_API) {
  realApi.checkHealth().then(isHealthy => {
    backendAvailable = isHealthy;
    if (isHealthy) {
      console.log('✓ Connected to real backend API at http://localhost:5001');
    } else {
      console.warn('⚠ Backend not available, using mock data');
    }
  });
}

export const api = {
  login: async (email: string) => {
    await new Promise(resolve => setTimeout(resolve, 800));
    return { id: 'u1', name: 'Ranger Smith', email, avatarUrl: 'https://ui-avatars.com/api/?name=Ranger+Smith&background=0D8ABC&color=fff' };
  },

  getStats: async (): Promise<DashboardStats> => {
    if (USE_REAL_API && backendAvailable) {
      try {
        return await realApi.getStats();
      } catch (error) {
        console.error('Failed to fetch stats from backend:', error);
      }
    }

    await new Promise(resolve => setTimeout(resolve, 600));
    return {
      totalZebras: MOCK_ZEBRAS.length,
      newToday: 3,
      totalSightings: 1245,
      activeCameras: 8,
      sightingsHistory: Array.from({ length: 7 }).map((_, i) => ({
        date: new Date(Date.now() - (6 - i) * 86400000).toLocaleDateString('en-US', { weekday: 'short' }),
        count: Math.floor(Math.random() * 30) + 10,
      })),
    };
  },

  getZebras: async (filter?: string): Promise<Zebra[]> => {
    if (USE_REAL_API && backendAvailable) {
      try {
        const zebras = await realApi.getZebras();
        if (filter === 'new') {
          return zebras.filter(z => z.status === 'new');
        }
        return zebras;
      } catch (error) {
        console.error('Failed to fetch zebras from backend:', error);
      }
    }

    await new Promise(resolve => setTimeout(resolve, 600));
    let zebras = [...MOCK_ZEBRAS];
    if (filter === 'new') zebras = zebras.filter(z => z.status === 'new');
    return zebras.sort((a, b) => new Date(b.lastSeen).getTime() - new Date(a.lastSeen).getTime());
  },

  getZebra: async (id: string): Promise<Zebra | undefined> => {
    if (USE_REAL_API && backendAvailable) {
      try {
        return await realApi.getZebra(id);
      } catch (error) {
        console.error('Failed to fetch zebra from backend:', error);
      }
    }

    await new Promise(resolve => setTimeout(resolve, 400));
    return MOCK_ZEBRAS.find(z => z.id === id);
  },

  getSightings: async (zebraId?: string): Promise<Sighting[]> => {
    await new Promise(resolve => setTimeout(resolve, 500));
    let sightings = [...MOCK_SIGHTINGS];
    if (zebraId) sightings = sightings.filter(s => s.zebraId === zebraId);
    return sightings.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
  },

  identifyZebra: async (file: File): Promise<{ zebra: Zebra; sighting: Sighting; isNew: boolean }> => {
    if (USE_REAL_API && backendAvailable) {
      try {
        return await realApi.identifyZebra(file);
      } catch (error) {
        console.error('Failed to identify zebra with backend:', error);
        // Fall through to mock implementation
      }
    }

    await new Promise(resolve => setTimeout(resolve, 2000)); // Simulate processing

    // Simulate zebra detection - 20% chance of no zebra detected
    const noZebraDetected = Math.random() < 0.2;

    if (noZebraDetected) {
      throw new Error('NO_ZEBRA_DETECTED');
    }

    const isNew = Math.random() > 0.7; // 30% chance of being a new zebra
    const imageUrl = URL.createObjectURL(file);

    let zebra: Zebra;

    if (isNew) {
      // Create a NEW zebra and add it to the database
      const newId = `ZEB-${1000 + MOCK_ZEBRAS.length}`;
      zebra = {
        id: newId,
        name: `Zebra ${1000 + MOCK_ZEBRAS.length}`,
        status: 'new',
        firstSeen: new Date().toISOString(),
        lastSeen: new Date().toISOString(),
        sightingsCount: 1,
        thumbnailUrl: imageUrl,
        tags: ['Upload'],
      };

      // Add to the beginning of the array so it appears first
      MOCK_ZEBRAS.unshift(zebra);
      console.log('New zebra added to database:', zebra.id);
    } else {
      // Match with an existing zebra
      const existingZebra = MOCK_ZEBRAS[Math.floor(Math.random() * Math.min(5, MOCK_ZEBRAS.length))];
      zebra = {
        ...existingZebra,
        lastSeen: new Date().toISOString(),
        sightingsCount: existingZebra.sightingsCount + 1,
        status: 'known',
      };

      // Update the zebra in the database
      const index = MOCK_ZEBRAS.findIndex(z => z.id === zebra.id);
      if (index !== -1) {
        MOCK_ZEBRAS[index] = zebra;
      }
      console.log('Existing zebra updated:', zebra.id);
    }

    // Create a new sighting
    const sighting: Sighting = {
      id: `SIGHT-${Date.now()}`,
      zebraId: zebra.id,
      timestamp: new Date().toISOString(),
      location: 'Upload',
      imageUrl: imageUrl,
      confidence: 0.88 + Math.random() * 0.11,
    };

    // Add sighting to database
    MOCK_SIGHTINGS.unshift(sighting);
    console.log('New sighting added:', sighting.id);

    return {
      zebra,
      isNew,
      sighting,
    };
  },

  clearAllZebras: async (): Promise<void> => {
    await new Promise(resolve => setTimeout(resolve, 500));
    MOCK_ZEBRAS = [];
    MOCK_SIGHTINGS = [];
    console.log('All zebras and sightings cleared from database');
  }
};
