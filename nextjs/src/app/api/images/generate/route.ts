// src/app/api/images/generate/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { getToken } from 'next-auth/jwt';
import { createClient } from '@supabase/supabase-js';

interface Txt2ImgRequestBody {
  user_id: string;
  bodyPrompt: string;
  facePrompt: string;
  width: number;
  height: number;
  upscaleEnabled: boolean;
  upscaleFactor: 2 | 4;
}

interface GenerateResponse {
  job_id: string;
  status: string;
  width?: number;
  height?: number;
  image_urls?: {
    final_image: string;
  };
  error?: string;
}

const MODAL_API_URL = process.env.MODAL_API_URL!;
const SUPABASE_URL = process.env.SUPABASE_URL!;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY!; // Use Service Role Key

// Initialize Supabase Client with Service Role Key
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY);

export async function POST(req: NextRequest) {
  try {
    // Get the token from NextAuth.js
    const token = await getToken({ req, secret: process.env.NEXTAUTH_SECRET });

    if (!token) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    // Extract user_id from the token
    const userId = token.sub as string;

    // Parse the request body
    const body: Txt2ImgRequestBody = await req.json();

    const { bodyPrompt, facePrompt, width, height, upscaleEnabled, upscaleFactor } = body;

    if (!bodyPrompt || !facePrompt || !width || !height) {
      return NextResponse.json(
        { error: 'bodyPrompt, facePrompt, and res are required.' },
        { status: 400 }
      );
    }

    // Prepare the payload for the Modal API, including user_id
    const modalPayload = {
      user_id: userId, // Include user_id from token
      image_prompt: bodyPrompt,
      face_prompt: facePrompt,
      width : width,
      height : height,
      upscale_enabled: upscaleEnabled,
      scaling: upscaleFactor,
    };

    // Submit the job to the Modal backend
    const modalResponse = await fetch(`${MODAL_API_URL}/generate-txt2img`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(modalPayload),
    });

    if (!modalResponse.ok) {
      const errorData = await modalResponse.json();
      console.error('Error from Modal API:', errorData);
      return NextResponse.json(
        { error: errorData.detail || 'Failed to generate image.' },
        { status: modalResponse.status }
      );
    }

    const data: GenerateResponse = await modalResponse.json();

    const { job_id, status } = data;

    if (status === 'queued') {
      // Since API routes cannot maintain state or set intervals,
      // the polling should be handled on the client-side.
      // The API route can respond with the job_id and status.
      return NextResponse.json({ job_id, status }, { status: 200 });
    } else {
      return NextResponse.json({ error: 'Unexpected job status.' }, { status: 400 });
    }
  } catch (error) {
    console.error('Error in /api/images/generate:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
