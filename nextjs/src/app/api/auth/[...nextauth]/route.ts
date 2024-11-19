// src/app/api/auth/[...nextauth]/route.ts

import NextAuth from 'next-auth';
import CredentialsProvider from 'next-auth/providers/credentials';
import { supabaseClient } from '../../../../lib/supabaseClient';

const handler = NextAuth({
  providers: [
    CredentialsProvider({
      name: 'Credentials',
      credentials: {
        email: { label: 'Email', type: 'email', placeholder: 'you@example.com' },
        password: { label: 'Password', type: 'password' },
      },
      async authorize(credentials) {
        const { email, password } = credentials as {
          email: string;
          password: string;
        };

        try {
          // Attempt to sign in the user with Supabase
          const { data, error } = await supabaseClient.auth.signInWithPassword({
            email,
            password,
          });

          if (error || !data.user) {
            console.error('Supabase signIn error:', error);
            throw new Error('Invalid email or password.');
          }

          // Check if email is confirmed
          const { data: userData, error: userError } = await supabaseClient.auth.getUser();

          if (userError) {
            console.error('Error fetching user data:', userError);
            throw new Error('Failed to retrieve user data.');
          }

          if (!userData.user?.email_confirmed_at) {
            throw new Error('Please confirm your email before logging in.');
          }

          // Return the user object as expected by NextAuth.js
          return {
            id: data.user.id,
            email: data.user.email,
            // Add any other user properties you want to include
          };
        } catch (error) {
          console.error('Authorize error:', error);
          throw new Error(error.message || 'Invalid email or password.');
        }
      },
    }),
    // Add other providers here if needed (e.g., Google, GitHub)
  ],
  session: {
    strategy: 'jwt',
  },
  callbacks: {
    async jwt({ token, user }) {
      if (user) {
        token.id = user.id;
      }
      return token;
    },
    async session({ session, token }) {
      if (token && session.user) {
        session.user.id = token.id as string;
      }
      return session;
    },
  },
  pages: {
    signIn: '/', // Redirect to landing page for sign in
    error: '/', // Redirect to landing page on error
  },
  secret: process.env.NEXTAUTH_SECRET,
});

export { handler as GET, handler as POST };
