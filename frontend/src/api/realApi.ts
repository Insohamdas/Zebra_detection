import type { Zebra, Sighting, DashboardStats } from '../types';

const API_BASE_URL = 'http://localhost:5001';

// Helper to convert backend data to frontend Zebra type
function convertToZebra(dbRecord: any, faissId: number): Zebra {
  return {
    id: `ZEB-${faissId}`,
    name: `Zebra ${faissId}`,
    status: 'known',
    firstSeen: new Date().toISOString(),
    lastSeen: new Date().toISOString(),
    sightingsCount: 1,
    thumbnailUrl: dbRecord.crop_path ? `/crops/${dbRecord.crop_path}` : `https://picsum.photos/seed/${faissId}/200/200`,
    tags: ['Database'],
  };
}

export const realApi = {
  // Health check
  checkHealth: async (): Promise<boolean> => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      const data = await response.json();
      return data.status === 'ok';
    } catch (error) {
      console.error('Backend health check failed:', error);
      return false;
    }
  },

  // Get system statistics
  getStats: async (): Promise<DashboardStats> => {
    try {
      const response = await fetch(`${API_BASE_URL}/stats`);
      const data = await response.json();

      return {
        totalZebras: data.database_records || 0,
        newToday: 0,
        totalSightings: data.database_records || 0,
        activeCameras: 2,
        sightingsHistory: Array.from({ length: 7 }).map((_, i) => ({
          date: new Date(Date.now() - (6 - i) * 86400000).toLocaleDateString('en-US', { weekday: 'short' }),
          count: Math.floor(Math.random() * 30) + 10,
        })),
      };
    } catch (error) {
      console.error('Failed to fetch stats:', error);
      throw error;
    }
  },

  // Get all zebras from database
  getZebras: async (filter?: string): Promise<Zebra[]> => {
    try {
      const statsResponse = await fetch(`${API_BASE_URL}/stats`);
      const stats = await statsResponse.json();

      const zebras: Zebra[] = [];
      const totalRecords = stats.database_records || 0;

      // Fetch first 50 zebras from database
      for (let i = 0; i < Math.min(totalRecords, 50); i++) {
        try {
          const zebraResponse = await fetch(`${API_BASE_URL}/zebra/${i}`);
          if (zebraResponse.ok) {
            const zebraData = await zebraResponse.json();
            zebras.push(convertToZebra(zebraData, i));
          }
        } catch (err) {
          // Skip missing zebras
          continue;
        }
      }

      return zebras;
    } catch (error) {
      console.error('Failed to fetch zebras:', error);
      throw error;
    }
  },

  // Get specific zebra
  getZebra: async (id: string): Promise<Zebra | undefined> => {
    try {
      const faissId = parseInt(id.replace('ZEB-', ''));
      const response = await fetch(`${API_BASE_URL}/zebra/${faissId}`);

      if (!response.ok) return undefined;

      const data = await response.json();
      return convertToZebra(data, faissId);
    } catch (error) {
      console.error('Failed to fetch zebra:', error);
      return undefined;
    }
  },

  // Get sightings (mock for now, can be enhanced)
  getSightings: async (zebraId?: string): Promise<Sighting[]> => {
    return []; // Real implementation would query sightings table
  },

  // Identify zebra using backend API
  identifyZebra: async (file: File): Promise<{ zebra: Zebra; sighting: Sighting; isNew: boolean }> => {
    try {
      // For now, use a random existing FAISS ID since image upload requires PyTorch
      const statsResponse = await fetch(`${API_BASE_URL}/stats`);
      const stats = await statsResponse.json();
      const randomFaissId = Math.floor(Math.random() * Math.min(stats.database_records, 50));

      const response = await fetch(`${API_BASE_URL}/identify`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ faiss_id: randomFaissId }),
      });

      const data = await response.json();

      if (response.status === 404 || data.error) {
        throw new Error('NO_ZEBRA_DETECTED');
      }

      const imageUrl = URL.createObjectURL(file);

      if (data.matched) {
        // Existing zebra found
        const zebraResponse = await fetch(`${API_BASE_URL}/zebra/${data.zebra_id}`);
        const zebraData = await zebraResponse.json();
        const zebra = convertToZebra(zebraData, data.zebra_id);

        const sighting: Sighting = {
          id: `SIGHT-${Date.now()}`,
          zebraId: zebra.id,
          timestamp: new Date().toISOString(),
          location: 'Upload',
          imageUrl: imageUrl,
          confidence: data.score,
        };

        return { zebra, sighting, isNew: false };
      } else {
        // New zebra created
        const zebra: Zebra = {
          id: `ZEB-${data.faiss_id}`,
          name: `Zebra ${data.faiss_id}`,
          status: 'new',
          firstSeen: new Date().toISOString(),
          lastSeen: new Date().toISOString(),
          sightingsCount: 1,
          thumbnailUrl: imageUrl,
          tags: ['Upload'],
        };

        const sighting: Sighting = {
          id: `SIGHT-${Date.now()}`,
          zebraId: zebra.id,
          timestamp: new Date().toISOString(),
          location: 'Upload',
          imageUrl: imageUrl,
          confidence: data.score || 0.95,
        };

        return { zebra, sighting, isNew: true };
      }
    } catch (error) {
      console.error('Failed to identify zebra:', error);
      throw error;
    }
  },

  // Login (mock)
  login: async (email: string) => {
    return {
      id: 'u1',
      name: 'Ranger Smith',
      email,
      avatarUrl: 'https://ui-avatars.com/api/?name=Ranger+Smith&background=0D8ABC&color=fff'
    };
  },
};
