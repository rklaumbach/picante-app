const express = require('express');
const router = express.Router();
const imageController = require('../controllers/imageController');
const auth = require('../middleware/auth');

router.post('/generate', auth, imageController.generateImage);
router.delete('/:id', auth, imageController.deleteImage);
router.get('/user', auth, imageController.getUserImages);

module.exports = router;
