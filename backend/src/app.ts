import dotenv from 'dotenv';
dotenv.config();

import express, { Request, Response, NextFunction } from 'express';
import cors from 'cors';
import connectDB from './database/connection';
import authRoutes from './routes/authRoutes';
import imageRoutes from './routes/imageRoutes';
import subscriptionRoutes from './routes/subscriptionRoutes';
import userRoutes from './routes/userRoutes';
import webhookRoutes from './routes/webhookRoutes';

const app = express();

// Connect to MongoDB
connectDB();

// Define allowed origins
const allowedOrigins = ['http://localhost:3000']; // Replace with your frontend URL

// Configure CORS middleware
app.use(
  cors({
    origin: allowedOrigins,
    credentials: true, // Allow cookies and auth headers
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization'],
  })
);

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Routes
app.use('/api/auth', authRoutes);
app.use('/api/images', imageRoutes);
app.use('/api/subscription', subscriptionRoutes);
app.use('/api/user', userRoutes);
app.use('/webhook', webhookRoutes);

// Error handling middleware
app.use((err: Error, req: Request, res: Response, next: NextFunction) => {
  console.error(err.stack);
  res.status(500).send('Something broke!');
});

// Start the server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});

export default app;
