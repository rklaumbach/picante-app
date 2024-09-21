// backend/src/routes/userRoutes.ts

import express from 'express';
import auth from '../middleware/auth';
import { getUserProfile, unsubscribe } from '../controllers/userController';

const router = express.Router();

router.get('/profile', auth, getUserProfile);
router.post('/unsubscribe', auth, unsubscribe);

export default router;
