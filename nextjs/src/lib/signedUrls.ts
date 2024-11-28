// src/lib/signedUrls.ts

import { supabaseClient } from './supabaseClient';

export const getSignedUrl = async (path: string, expiresIn: number = 60 * 60) => { // default 1 hour
  const { data, error } = await supabaseClient
    .storage
    .from('character-images')
    .createSignedUrl(path, expiresIn);

  if (error) {
    console.error('Error generating signed URL:', error);
    return '';
  }

  return data.signedUrl;
};
