"use strict";
// backend/src/controllers/userController.ts
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.updatePassword = exports.unsubscribe = exports.getUserProfile = void 0;
const User_1 = __importDefault(require("../models/User"));
const stripe_1 = __importDefault(require("stripe"));
const bcryptjs_1 = __importDefault(require("bcryptjs"));
const stripe = new stripe_1.default(process.env.STRIPE_SECRET_KEY, { apiVersion: '2020-08-27' });
const getUserProfile = async (req, res) => {
    const userId = req.user.userId;
    try {
        const user = await User_1.default.findById(userId, 'email subscriptionStatus');
        if (!user) {
            return res.status(404).json({ error: 'User not found' });
        }
        res.json({ email: user.email, subscriptionStatus: user.subscriptionStatus });
    }
    catch (error) {
        console.error('Error fetching user profile:', error);
        res.status(500).json({ error: 'Failed to fetch user profile' });
    }
};
exports.getUserProfile = getUserProfile;
const unsubscribe = async (req, res) => {
    const userId = req.user.userId;
    const { reasons, additionalFeedback } = req.body;
    try {
        const user = await User_1.default.findById(userId);
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
    }
    catch (error) {
        console.error('Error cancelling subscription:', error);
        res.status(500).json({ error: 'Failed to cancel subscription' });
    }
};
exports.unsubscribe = unsubscribe;
const updatePassword = async (req, res) => {
    const userId = req.user.userId;
    const { currentPassword, newPassword } = req.body;
    try {
        const user = await User_1.default.findById(userId);
        if (!user) {
            return res.status(404).json({ error: 'User not found' });
        }
        // Verify current password
        const isMatch = await bcryptjs_1.default.compare(currentPassword, user.password);
        if (!isMatch) {
            return res.status(400).json({ error: 'Current password is incorrect' });
        }
        // Hash new password
        const hashedPassword = await bcryptjs_1.default.hash(newPassword, 10);
        // Update user's password
        user.password = hashedPassword;
        await user.save();
        res.json({ message: 'Password updated successfully' });
    }
    catch (error) {
        console.error('Error updating password:', error);
        res.status(500).json({ error: 'Failed to update password' });
    }
};
exports.updatePassword = updatePassword;
//# sourceMappingURL=userController.js.map