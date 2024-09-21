"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const jsonwebtoken_1 = __importDefault(require("jsonwebtoken"));
const auth = (req, res, next) => {
    const authHeader = req.header('Authorization');
    if (!authHeader)
        return res.status(401).json({ error: 'Access denied. No token provided.' });
    const token = authHeader.replace('Bearer ', '');
    try {
        const decoded = jsonwebtoken_1.default.verify(token, process.env.JWT_SECRET);
        req.user = decoded;
        next();
    }
    catch (ex) {
        res.status(400).json({ error: 'Invalid token.' });
    }
};
exports.default = auth;
//# sourceMappingURL=auth.js.map