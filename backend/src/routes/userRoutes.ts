// src/routes/userRoutes.ts

import express from 'express';
import auth from '../middleware/auth';
import { getUserProfile, unsubscribe, updatePassword } from '../controllers/userController';

const router = express.Router();

router.get('/profile', auth, getUserProfile);
router.post('/unsubscribe', auth, unsubscribe);
router.put('/update-password', auth, updatePassword); // Add this line

export default router;
