require('dotenv').config();
const express = require('express');
const bodyParser = require('body-parser');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const cors = require('cors');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(bodyParser.json({ limit: "50mb" }));
app.use(bodyParser.urlencoded({ extended: true }));

// Initialize Google Generative AI
const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
  console.error('GEMINI_API_KEY is not set in environment variables');
  process.exit(1);
}

const genAI = new GoogleGenerativeAI(apiKey);

const model = genAI.getGenerativeModel({
  model: "gemini-1.5-flash", // Updated to current model name
});

const generationConfig = {
  temperature: 1,
  topP: 0.95,
  topK: 40,
  maxOutputTokens: 8192,
  responseMimeType: "text/plain",
};

// Set up Multer for image uploads
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    const uploadDir = 'uploads/';
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    if (ext !== '.jpg' && ext !== '.jpeg' && ext !== '.png') {
      return cb(new Error('Only .jpg, .jpeg or .png files are allowed.'));
    }
    cb(null, Date.now() + ext);
  }
});

const upload = multer({ 
  storage: storage,
  limits: {
    fileSize: 5 * 1024 * 1024 // 5MB file size limit
  }
}).single('image');

// Endpoint for AI response
app.post('/ai-response', async (req, res) => {
  const { prompt } = req.body;

  if (!prompt) {
    return res.status(400).json({ error: 'Prompt is required.' });
  }

  try {
    const chatSession = model.startChat({
      generationConfig,
      history: [
        {
          role: "user",
          parts: [
            { text: "You are a farming AI assistant named Zaytoon AI. Provide helpful, accurate information about agricultural topics." },
          ],
        },
        // Add more history if needed
      ],
    });

    const result = await chatSession.sendMessage(prompt);
    const response = await result.response;
    res.json({ response: response.text() });
  } catch (error) {
    console.error('Error generating AI response:', error);
    res.status(500).json({ error: 'An error occurred while processing your request.' });
  }
});

// Image and AI Response Endpoint
app.post("/ai-response-with-image", (req, res) => {
  upload(req, res, async (err) => {
    if (err) {
      if (err instanceof multer.MulterError) {
        return res.status(400).json({ error: err.message });
      } else {
        return res.status(400).json({ error: err.message });
      }
    }

    if (!req.file) {
      return res.status(400).json({ error: "No image uploaded." });
    }

    const { prompt } = req.body;

    if (!prompt) {
      // Clean up uploaded file if prompt is missing
      if (req.file) {
        fs.unlinkSync(req.file.path);
      }
      return res.status(400).json({ error: "Prompt is required." });
    }

    try {
      // Read the uploaded image and convert it to base64
      const imagePath = path.join(__dirname, req.file.path);
      const imageBuffer = fs.readFileSync(imagePath);
      const imageBase64 = imageBuffer.toString('base64');
      
      // Call the AI model with the image data
      const result = await model.generateContent([
        {
          inlineData: {
            data: imageBase64,
            mimeType: `image/${path.extname(req.file.filename).substring(1)}`,
          },
        },
        { text: prompt },
      ]);

      const response = await result.response;
      
      // Clean up uploaded image after processing
      fs.unlinkSync(imagePath);

      res.json({ response: response.text() });
    } catch (error) {
      console.error("Error handling AI response with image:", error);
      // Clean up uploaded file if error occurs
      if (req.file) {
        fs.unlinkSync(req.file.path);
      }
      res.status(500).json({ error: "Failed to generate response with image." });
    }
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: 'Something went wrong!' });
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});