"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const imageController_1 = require("../controllers/imageController");
const auth_1 = __importDefault(require("../middleware/auth"));
const router = express_1.default.Router();
router.post('/generate', auth_1.default, imageController_1.generateImage);
router.delete('/:id', auth_1.default, imageController_1.deleteImage);
router.get('/user', auth_1.default, imageController_1.getUserImages);
exports.default = router;
//# sourceMappingURL=imageRoutes.js.map