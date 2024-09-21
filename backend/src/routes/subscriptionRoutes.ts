import express from 'express';
import auth from '../middleware/auth';
import { createSubscription } from '../controllers/subscriptionController';

const router = express.Router();

router.post('/create-subscription', auth, createSubscription);

export default router;
