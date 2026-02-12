export interface Zebra {
  id: string;
  name: string; // e.g. "Zebra 123"
  status: 'new' | 'known';
  firstSeen: string; // ISO date
  lastSeen: string; // ISO date
  sightingsCount: number;
  thumbnailUrl: string;
  notes?: string;
  tags?: string[];
}

export interface Sighting {
  id: string;
  zebraId: string;
  timestamp: string;
  location: string;
  imageUrl: string;
  confidence: number;
}

export interface User {
  id: string;
  name: string;
  email: string;
  avatarUrl?: string;
}

export interface DashboardStats {
  totalZebras: number;
  newToday: number;
  totalSightings: number;
  activeCameras: number;
  sightingsHistory: { date: string; count: number }[];
}
