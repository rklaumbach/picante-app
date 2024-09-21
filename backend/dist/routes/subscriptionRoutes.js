"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const auth_1 = __importDefault(require("../middleware/auth"));
const subscriptionController_1 = require("../controllers/subscriptionController");
const router = express_1.default.Router();
router.post('/create-subscription', auth_1.default, subscriptionController_1.createSubscription);
exports.default = router;
//# sourceMappingURL=subscriptionRoutes.js.map