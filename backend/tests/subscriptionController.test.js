// backend/tests/subscriptionController.test.js

const request = require('supertest');
const app = require('../src/app');
const mongoose = require('mongoose');
const User = require('../src/models/User');

describe('Subscription Controller', () => {
  let token;
  let userId;

  beforeAll(async () => {
    await mongoose.connect(process.env.MONGODB_URI_TEST);

    const user = new User({ email: 'subscriber@example.com', password: 'hashedpassword' });
    await user.save();

    userId = user._id;
    token = 'your_test_jwt_token';
  });

  afterAll(async () => {
    await User.deleteMany({});
    await mongoose.disconnect();
  });

  test('Create subscription successfully', async () => {
    // Mock Stripe API
    jest.mock('stripe', () => {
      return jest.fn().mockImplementation(() => ({
        customers: {
          create: jest.fn().mockResolvedValue({ id: 'cus_test' }),
        },
        subscriptions: {
          create: jest.fn().mockResolvedValue({ id: 'sub_test' }),
        },
      }));
    });

    const response = await request(app)
      .post('/api/subscription/create-subscription')
      .set('Authorization', `Bearer ${token}`)
      .send({ paymentMethodId: 'pm_test' });

    expect(response.statusCode).toBe(200);
    expect(response.body).toHaveProperty('message', 'Subscription created successfully');
  });
});
