// backend/tests/imageController.test.js

const request = require('supertest');
const app = require('../src/app');
const mongoose = require('mongoose');
const User = require('../src/models/User');

describe('Image Controller', () => {
  let token;
  let userId;

  beforeAll(async () => {
    // Connect to test database and create a test user
    await mongoose.connect(process.env.MONGODB_URI_TEST);

    const user = new User({ email: 'testuser@example.com', password: 'hashedpassword', credits: 5 });
    await user.save();

    userId = user._id;
    token = 'your_test_jwt_token'; // Generate a test token
  });

  afterAll(async () => {
    await User.deleteMany({});
    await mongoose.disconnect();
  });

  test('Generate image with sufficient credits', async () => {
    // Mock comfyuiService
    jest.mock('../src/services/comfyuiService', () => ({
      generateImage: jest.fn().mockResolvedValue({ imageUrl: 'http://example.com/image.jpg' }),
    }));

    const response = await request(app)
      .post('/api/images/generate')
      .set('Authorization', `Bearer ${token}`)
      .send({ bodyPrompt: 'Test body', facePrompt: 'Test face' });

    expect(response.statusCode).toBe(200);
    expect(response.body).toHaveProperty('imageUrl');
  });
});
