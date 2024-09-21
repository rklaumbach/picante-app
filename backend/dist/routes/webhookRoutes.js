"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const stripe_1 = __importDefault(require("stripe"));
const User_1 = __importDefault(require("../models/User"));
const router = express_1.default.Router();
const stripe = new stripe_1.default(process.env.STRIPE_SECRET_KEY, { apiVersion: '2020-08-27' });
router.post('/stripe', express_1.default.raw({ type: 'application/json' }), async (req, res) => {
    const sig = req.headers['stripe-signature'];
    const endpointSecret = process.env.STRIPE_WEBHOOK_SECRET;
    let event;
    try {
        event = stripe.webhooks.constructEvent(req.body, sig, endpointSecret);
    }
    catch (err) {
        console.error('Webhook signature verification failed:', err.message);
        return res.status(400).send(`Webhook Error: ${err.message}`);
    }
    const dataObject = event.data.object;
    // Handle the event
    switch (event.type) {
        case 'customer.subscription.updated':
        case 'customer.subscription.deleted':
            const customerId = dataObject.customer;
            const subscriptionStatus = dataObject.status;
            // Update user's subscription status in the database
            await User_1.default.findOneAndUpdate({ stripeCustomerId: customerId }, { subscriptionStatus });
            break;
        // ... handle other event types
        default:
            console.log(`Unhandled event type ${event.type}`);
    }
    res.json({ received: true });
});
exports.default = router;
//# sourceMappingURL=webhookRoutes.js.map