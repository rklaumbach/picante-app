import express from 'express';
import { generateImage, deleteImage, getUserImages } from '../controllers/imageController';
import auth from '../middleware/auth';

const router = express.Router();

router.post('/generate', auth, generateImage);
router.delete('/:id', auth, deleteImage);
router.get('/user', auth, getUserImages);

export default router;
