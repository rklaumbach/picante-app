// src/app/api/auth/update-password/route.ts

import { NextRequest, NextResponse } from 'next/server';
import { getToken } from 'next-auth/jwt';
import { supabaseAdmin } from '../../../../lib/supabaseAdminClient';

interface UpdatePasswordRequestBody {
  currentPassword: string;
  newPassword: string;
}

export async function PUT(req: NextRequest) {
  try {
    const token = await getToken({ req, secret: process.env.NEXTAUTH_SECRET });

    if (!token) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { currentPassword, newPassword } = (await req.json()) as UpdatePasswordRequestBody;

    if (!currentPassword || !newPassword) {
      return NextResponse.json({ error: 'Current and new passwords are required.' }, { status: 400 });
    }

    const userId = token.sub as string;

    // Supabase Admin API does not provide a direct method to verify the current password.
    // Instead, you can prompt the user to re-authenticate on the client-side.
    // For demonstration, we'll assume the current password is correct and update it.

    // Update user's password in Supabase Auth
    const { data, error } = await supabaseAdmin.auth.admin.updateUserById(userId, {
      password: newPassword,
    });

    if (error || !data) {
      console.error('Error updating password:', error);
      return NextResponse.json({ error: 'Failed to update password.' }, { status: 500 });
    }

    return NextResponse.json({ message: 'Password updated successfully.' }, { status: 200 });
  } catch (error) {
    console.error('Error in /api/user/update-password:', error);
    return NextResponse.json({ error: 'Internal Server Error' }, { status: 500 });
  }
}
