// src/types/types.ts

export interface ImageData {
  id: string;
  image_url: string;
  filename: string;
  body_prompt: string;
  face_prompt: string;
  width: number;
  height: number;
  created_at: string;
}

export interface Character {
  id: string;
  user_id: string;
  name: string;
  image_id: string; // Changed from image_path to image_id
  signed_image_url?: string;
  personality_traits: string;
  other_info: string;
  created_at: string;
  updated_at: string;
}

export interface Chat {
  id: string;
  user_id: string;
  title: string;
  scenario: string;
  character_id: string;
  last_response: string;
  updated_at: string;
  character: Character; // Added property
}

export interface Message {
  id: string;
  chat_id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

export interface SupabaseRealtimePayload<T> {
  eventType: string;
  new: T;
  old: T | null;
  schema: string;
  table: string;
  commit_timestamp: string;
}