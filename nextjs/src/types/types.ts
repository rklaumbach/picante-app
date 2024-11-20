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
