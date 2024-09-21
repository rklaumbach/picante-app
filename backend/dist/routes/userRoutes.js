"use strict";
// backend/src/routes/userRoutes.ts
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const auth_1 = __importDefault(require("../middleware/auth"));
const userController_1 = require("../controllers/userController");
const router = express_1.default.Router();
router.get('/profile', auth_1.default, userController_1.getUserProfile);
router.post('/unsubscribe', auth_1.default, userController_1.unsubscribe);
exports.default = router;
//# sourceMappingURL=userRoutes.js.map