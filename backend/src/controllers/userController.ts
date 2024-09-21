// backend/src/controllers/userController.ts

import { Response } from 'express';
import User from '../models/User';
import Stripe from 'stripe';
import { AuthenticatedRequest } from '../middleware/auth';

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY!, { apiVersion: '2020-08-27' });

export const getUserProfile = async (req: AuthenticatedRequest, res: Response) => {
  const userId = req.user.userId;

  try {
    const user = await User.findById(userId, 'email subscriptionStatus');
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }
    res.json({ email: user.email, subscriptionStatus: user.subscriptionStatus });
  } catch (error) {
    console.error('Error fetching user profile:', error);
    res.status(500).json({ error: 'Failed to fetch user profile' });
  }
};

export const unsubscribe = async (req: AuthenticatedRequest, res: Response) => {
  const userId = req.user.userId;
  const { reasons, additionalFeedback } = req.body;

  try {
    const user = await User.findById(userId);
    if (!user || !user.stripeCustomerId) {
      return res.status(404).json({ error: 'User not found or not subscribed' });
    }

    // Retrieve the active subscription
    const subscriptions = await stripe.subscriptions.list({
      customer: user.stripeCustomerId,
      status: 'active',
      limit: 1,
    });

    if (subscriptions.data.length === 0) {
      return res.status(400).json({ error: 'No active subscriptions found' });
    }

    const subscriptionId = subscriptions.data[0].id;

    // Cancel the subscription
    await stripe.subscriptions.del(subscriptionId);

    // Update user's subscription status
    user.subscriptionStatus = 'cancelled';
    await user.save();

    // Optionally, handle the feedback
    // For example, save to the database or send an email

    res.json({ message: 'Subscription cancelled successfully' });
  } catch (error) {
    console.error('Error cancelling subscription:', error);
    res.status(500).json({ error: 'Failed to cancel subscription' });
  }
};
