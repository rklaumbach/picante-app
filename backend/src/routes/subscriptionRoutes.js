const express = require('express');
const router = express.Router();
const auth = require('../middleware/auth');
const subscriptionController = require('../controllers/subscriptionController');

router.post('/create-subscription', auth, subscriptionController.createSubscription);

module.exports = router;
