"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const dotenv_1 = __importDefault(require("dotenv"));
dotenv_1.default.config();
const express_1 = __importDefault(require("express"));
const cors_1 = __importDefault(require("cors"));
const connection_1 = __importDefault(require("./database/connection"));
const authRoutes_1 = __importDefault(require("./routes/authRoutes"));
const imageRoutes_1 = __importDefault(require("./routes/imageRoutes"));
const subscriptionRoutes_1 = __importDefault(require("./routes/subscriptionRoutes"));
const userRoutes_1 = __importDefault(require("./routes/userRoutes"));
const webhookRoutes_1 = __importDefault(require("./routes/webhookRoutes"));
const app = (0, express_1.default)();
// Connect to MongoDB
(0, connection_1.default)();
// Middleware
app.use((0, cors_1.default)());
app.use(express_1.default.json());
app.use(express_1.default.urlencoded({ extended: true }));
// Routes
app.use('/api/auth', authRoutes_1.default);
app.use('/api/images', imageRoutes_1.default);
app.use('/api/subscription', subscriptionRoutes_1.default);
app.use('/api/user', userRoutes_1.default);
app.use('/webhook', webhookRoutes_1.default);
// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).send('Something broke!');
});
// Start the server
const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
exports.default = app;
//# sourceMappingURL=app.js.map