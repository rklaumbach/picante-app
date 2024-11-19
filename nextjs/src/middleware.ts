// src/middleware.ts

import { NextRequest, NextResponse } from 'next/server';
import { RateLimiterMemory } from 'rate-limiter-flexible';
import { getToken } from 'next-auth/jwt';

const JWT_SECRET = process.env.NEXTAUTH_SECRET!;

// Rate limiter setup
const protectedRouteRateLimiter = new RateLimiterMemory({
  points: 100, // Number of points
  duration: 900, // Per 900 seconds (15 minutes)
});

const jobStatusRateLimiter = new RateLimiterMemory({
  points: 1000, // Higher limit for job status
  duration: 900, // 15 minutes
});

// Function to extract client IP
const getClientIp = (req: NextRequest): string => {
  const xForwardedFor = req.headers.get("x-forwarded-for");
  if (xForwardedFor) {
    // 'x-forwarded-for' can contain multiple IPs, the first one is the client's IP
    const ips = xForwardedFor.split(",").map((ip) => ip.trim());
    return ips[0];
  }

  const cfConnectingIp = req.headers.get("cf-connecting-ip");
  if (cfConnectingIp) {
    return cfConnectingIp;
  }

  // Fallback if no IP is found
  return "unknown";
};

// Define protected routes
const protectedRoutes = [
  "/generate",
  "/account",
  "/gallery",
  "/fullview",
  "/api/images",
  "/api/auth/update-email",
  "/api/auth/update-password",
  // Add any other protected routes as needed
];

// Middleware function
export async function middleware(req: NextRequest) {
  const ip = getClientIp(req);
  const { pathname } = req.nextUrl;

  // Log all incoming cookies for debugging
  // console.log("Middleware: Incoming Cookies:", JSON.stringify(req.cookies.getAll()));

  // Apply rate limiting based on the route
  try {
    if (pathname.startsWith("/api/images")) {
      // Apply higher rate limit for job status and image-related API routes
      await jobStatusRateLimiter.consume(ip);
    } else {
      // Apply standard rate limit for other protected routes
      await protectedRouteRateLimiter.consume(ip);
    }
  } catch (err) {
    // Too Many Requests
    return new NextResponse("Too Many Requests", { status: 429 });
  }

  // Check if the requested route is protected
  if (protectedRoutes.some((route) => pathname.startsWith(route))) {
    // Retrieve token using NextAuth.js's getToken
    const token = await getToken({ req, secret: JWT_SECRET , secureCookie : true
    });

    // Log the retrieved token
    console.log("Middleware: Retrieved Token:", JSON.stringify(token));

    if (!token) {
      console.log("Middleware: No token found, redirecting to /");
      const url = req.nextUrl.clone();
      url.pathname = "/";
      return NextResponse.redirect(url);
    }

    console.log("Middleware: Token verified, payload:", token);
    return NextResponse.next();
  }

  // Handle redirecting authenticated users away from the landing page
  if (pathname === "/") {
    const token = await getToken({ req, secret: JWT_SECRET , secureCookie : true
    });

    // Log the retrieved token for the landing page
    console.log("Middleware: Retrieved Token for '/' route:", JSON.stringify(token));

    if (token) {
      console.log(
        "Middleware: Authenticated user accessing /, redirecting to /generate"
      );
      const url = req.nextUrl.clone();
      url.pathname = "/generate";
      return NextResponse.redirect(url);
    } else {
      console.log("Middleware: Unauthenticated user accessing /, allowing access");
      // Allow access to '/'
    }
  }

  return NextResponse.next();
}

// Configuration for middleware
export const config = {
  matcher: [
    "/",
    "/generate/:path*",
    "/account/:path*",
    "/gallery/:path*",
    "/fullview/:path*",
    "/api/images/:path*", // Protect API routes
    "/api/auth/update-email",
    "/api/auth/update-password",
  ],
};
