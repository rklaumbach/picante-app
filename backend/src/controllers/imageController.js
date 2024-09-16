const comfyuiService = require('../services/comfyuiService');
const Image = require('../models/Image');
const User = require('../models/User');

exports.generateImage = async (req, res) => {
  const { bodyPrompt, facePrompt } = req.body;
  const userId = req.user.userId;

  try {
    const user = await User.findById(userId);

    // Check if user has enough credits or is a subscriber
    if (user.subscriptionStatus !== 'active' && user.credits <= 0) {
      return res.status(403).json({ error: 'Out of credits' });
    }

    // Call the service to interact with ComfyUI API
    const imageResult = await comfyuiService.generateImage({ bodyPrompt, facePrompt });

    // Save image details to the database
    const newImage = new Image({
      userId,
      imageUrl: imageResult.imageUrl,
      bodyPrompt,
      facePrompt,
      resolution: imageResult.resolution,
    });

    await newImage.save();

    // Decrement user's credits if not a subscriber
    if (user.subscriptionStatus !== 'active') {
      user.credits -= 1;
      await user.save();
    }

    // Respond with the image URL
    res.status(200).json({ imageUrl: imageResult.imageUrl });
  } catch (error) {
    console.error('Error generating image:', error);
    res.status(500).json({ error: 'Failed to generate image' });
  }
};

exports.deleteImage = async (req, res) => {
  const imageId = req.params.id;
  const userId = req.user.userId;

  try {
    const image = await Image.findOne({ _id: imageId, userId });

    if (!image) {
      return res.status(404).json({ error: 'Image not found' });
    }

    await Image.deleteOne({ _id: imageId });

    res.status(200).json({ message: 'Image deleted successfully' });
  } catch (error) {
    console.error('Error deleting image:', error);
    res.status(500).json({ error: 'Failed to delete image' });
  }
};

exports.getUserImages = async (req, res) => {
  const userId = req.user.userId;

  try {
    const images = await Image.find({ userId }).sort({ timestamp: -1 });
    res.status(200).json({ images });
  } catch (error) {
    console.error('Error fetching images:', error);
    res.status(500).json({ error: 'Failed to fetch images' });
  }
};
