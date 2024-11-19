// src/app/api/images/job-status/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { getToken } from 'next-auth/jwt';

interface JobStatusResponse {
  status: string;
  image_urls?: Record<string, string>;
  reason?: string;
}

// Environment Variables
const MODAL_API_URL = process.env.MODAL_API_URL; // e.g., 'http://localhost:8000' for local development

if (!MODAL_API_URL) {
  throw new Error('MODAL_API_URL is not defined in environment variables.');
}

export async function GET(req: NextRequest) {
  try {
    console.log('GET /api/images/job-status called'); // Debugging log

    // Use getToken from next-auth for consistent token handling
    const token = await getToken({ req, secret: process.env.NEXTAUTH_SECRET });

    if (!token) {
      console.warn('Unauthorized access attempt');
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    // Extract user_id from the token
    const userId = token.sub as string;
    console.log('Authenticated user ID:', userId);

    // Get job_id from query parameters
    const { searchParams } = new URL(req.url);
    const job_id = searchParams.get('job_id');

    if (!job_id) {
      console.error('Missing job_id in query parameters');
      return NextResponse.json({ error: 'job_id query parameter is required.' }, { status: 400 });
    }

    // Optionally, verify that the job_id belongs to the userId
    // This depends on your backend's implementation

    // Query the Modal backend for job status
    const modalResponse = await fetch(`${MODAL_API_URL}/job-status/${job_id}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
        // Include Authorization header if Modal API requires it
        // 'Authorization': `Bearer YOUR_MODAL_API_KEY`,
      },
    });

    if (!modalResponse.ok) {
      const errorData = await modalResponse.json();
      console.error('Error from Modal API:', errorData);
      return NextResponse.json(
        { error: errorData.detail || 'Failed to fetch job status.' },
        { status: modalResponse.status }
      );
    }

    const data: JobStatusResponse = await modalResponse.json();

    console.log('Response from Modal API:', data); // Debugging log

    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error('Error in /api/images/job-status:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
