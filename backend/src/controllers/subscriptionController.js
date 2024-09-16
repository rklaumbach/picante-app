const stripe = require('stripe')(process.env.STRIPE_SECRET_KEY);
const User = require('../models/User');

exports.createSubscription = async (req, res) => {
  const { paymentMethodId } = req.body;
  const userId = req.user.userId;

  try {
    const user = await User.findById(userId);

    // Create or retrieve Stripe customer
    let customer;
    if (!user.stripeCustomerId) {
      customer = await stripe.customers.create({
        email: user.email,
        payment_method: paymentMethodId,
        invoice_settings: {
          default_payment_method: paymentMethodId,
        },
      });
      user.stripeCustomerId = customer.id;
      await user.save();
    } else {
      customer = await stripe.customers.retrieve(user.stripeCustomerId);
    }

    // Create subscription
    const subscription = await stripe.subscriptions.create({
      customer: customer.id,
      items: [{ price: process.env.STRIPE_PRICE_ID }],
      expand: ['latest_invoice.payment_intent'],
    });

    // Update user's subscription status
    user.subscriptionStatus = 'active';
    await user.save();

    res.status(200).json({ message: 'Subscription created successfully' });
  } catch (error) {
    console.error('Subscription error:', error);
    res.status(500).json({ error: 'Subscription creation failed' });
  }
};
