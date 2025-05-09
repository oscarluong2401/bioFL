import os
import random
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

class FingerprintCNN(nn.Module):
    def __init__(self):
        super(FingerprintCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # Increased to 16 channels
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # Increased to 32 channels
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),  # Increased to 48 channels
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),  # Increased to 64 channels
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(64 * 6 * 6, 256),  # Adjusted input size for 96x96 images
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 600)
        )

    def forward(self, x):
        x = self.net(x)
        return x

model = FingerprintCNN()
model.net.load_state_dict(torch.load("bioFL_trained/last.pth", map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((96, 96)),  # already 96x96, but safe
    transforms.ToTensor(),
])

# A mapping of the predicted class to the original class
# This is based on the classification of the validation set
# Hence it may differ depending on the training phase
class_map = {0: 1, 111: 2, 222: 3, 333: 4, 444: 5, 555: 6, 567: 7, 578: 8, 589: 9, 1: 10, 12: 11, 23: 12, 34: 13, 45: 14, 56: 15, 67: 16, 78: 17, 89: 18, 100: 19, 112: 20, 123: 21, 134: 22, 145: 23, 156: 24, 167: 25, 178: 26, 189: 27, 200: 28, 211: 29, 223: 30, 234: 31, 245: 32, 256: 33, 267: 34, 278: 35, 289: 36, 300: 37, 311: 38, 322: 39, 334: 40, 345: 41, 356: 42, 367: 43, 378: 44, 389: 45, 400: 46, 411: 47, 422: 48, 433: 49, 445: 50, 456: 51, 467: 52, 478: 53, 489: 54, 500: 55, 511: 56, 522: 57, 533: 58, 544: 59, 556: 60, 558: 61, 559: 62, 560: 63, 561: 64, 562: 65, 563: 66, 564: 67, 565: 68, 566: 69, 568: 70, 569: 71, 570: 72, 571: 73, 572: 74, 573: 75, 574: 76, 575: 77, 576: 78, 577: 79, 579: 80, 580: 81, 581: 82, 582: 83, 583: 84, 584: 85, 585: 86, 586: 87, 587: 88, 588: 89, 590: 90, 591: 91, 592: 92, 593: 93, 594: 94, 595: 95, 596: 96, 597: 97, 598: 98, 599: 99, 2: 100, 3: 101, 4: 102, 5: 103, 6: 104, 7: 105, 8: 106, 9: 107, 10: 108, 11: 109, 13: 110, 14: 111, 15: 112, 16: 113, 17: 114, 18: 115, 19: 116, 20: 117, 21: 118, 22: 119, 24: 120, 25: 121, 26: 122, 27: 123, 28: 124, 29: 125, 30: 126, 31: 127, 32: 128, 33: 129, 35: 130, 36: 131, 37: 132, 38: 133, 39: 134, 40: 135, 41: 136, 42: 137, 43: 138, 44: 139, 46: 140, 47: 141, 48: 142, 49: 143, 50: 144, 51: 145, 52: 146, 53: 147, 54: 148, 55: 149, 57: 150, 58: 151, 59: 152, 60: 153, 61: 154, 62: 155, 63: 156, 64: 157, 65: 158, 66: 159, 68: 160, 69: 161, 70: 162, 71: 163, 72: 164, 73: 165, 74: 166, 75: 167, 76: 168, 77: 169, 79: 170, 80: 171, 81: 172, 82: 173, 83: 174, 84: 175, 85: 176, 86: 177, 87: 178, 88: 179, 90: 180, 91: 181, 92: 182, 93: 183, 94: 184, 95: 185, 96: 186, 97: 187, 98: 188, 99: 189, 101: 190, 102: 191, 103: 192, 104: 193, 105: 194, 106: 195, 107: 196, 108: 197, 109: 198, 110: 199, 113: 200, 114: 201, 115: 202, 116: 203, 117: 204, 118: 205, 119: 206, 120: 207, 121: 208, 122: 209, 124: 210, 125: 211, 126: 212, 127: 213, 128: 214, 129: 215, 130: 216, 131: 217, 132: 218, 133: 219, 135: 220, 136: 221, 137: 222, 138: 223, 139: 224, 140: 225, 141: 226, 142: 227, 143: 228, 144: 229, 146: 230, 147: 231, 148: 232, 149: 233, 150: 234, 151: 235, 152: 236, 153: 237, 154: 238, 155: 239, 157: 240, 158: 241, 159: 242, 160: 243, 161: 244, 162: 245, 163: 246, 164: 247, 165: 248, 166: 249, 168: 250, 169: 251, 170: 252, 171: 253, 172: 254, 173: 255, 174: 256, 175: 257, 176: 258, 177: 259, 179: 260, 180: 261, 181: 262, 182: 263, 183: 264, 184: 265, 185: 266, 186: 267, 187: 268, 188: 269, 190: 270, 191: 271, 192: 272, 193: 273, 194: 274, 195: 275, 196: 276, 197: 277, 198: 278, 199: 279, 201: 280, 202: 281, 203: 282, 204: 283, 205: 284, 206: 285, 207: 286, 208: 287, 209: 288, 210: 289, 212: 290, 213: 291, 214: 292, 215: 293, 216: 294, 217: 295, 218: 296, 219: 297, 220: 298, 221: 299, 224: 300, 225: 301, 226: 302, 227: 303, 228: 304, 229: 305, 230: 306, 231: 307, 232: 308, 233: 309, 235: 310, 236: 311, 237: 312, 238: 313, 239: 314, 240: 315, 241: 316, 242: 317, 243: 318, 244: 319, 246: 320, 247: 321, 248: 322, 249: 323, 250: 324, 251: 325, 252: 326, 253: 327, 254: 328, 255: 329, 257: 330, 258: 331, 259: 332, 260: 333, 261: 334, 262: 335, 263: 336, 264: 337, 265: 338, 266: 339, 268: 340, 269: 341, 270: 342, 271: 343, 272: 344, 273: 345, 274: 346, 275: 347, 276: 348, 277: 349, 279: 350, 280: 351, 281: 352, 282: 353, 283: 354, 284: 355, 285: 356, 286: 357, 287: 358, 288: 359, 290: 360, 291: 361, 292: 362, 293: 363, 294: 364, 295: 365, 296: 366, 297: 367, 298: 368, 299: 369, 301: 370, 302: 371, 303: 372, 304: 373, 305: 374, 306: 375, 307: 376, 308: 377, 309: 378, 310: 379, 312: 380, 313: 381, 314: 382, 315: 383, 316: 384, 317: 385, 318: 386, 319: 387, 320: 388, 321: 389, 323: 390, 324: 391, 325: 392, 326: 393, 327: 394, 328: 395, 329: 396, 330: 397, 331: 398, 332: 399, 335: 400, 336: 401, 337: 402, 338: 403, 339: 404, 340: 405, 341: 406, 342: 407, 343: 408, 344: 409, 346: 410, 347: 411, 348: 412, 349: 413, 350: 414, 351: 415, 352: 416, 353: 417, 354: 418, 355: 419, 357: 420, 358: 421, 359: 422, 360: 423, 361: 424, 362: 425, 363: 426, 364: 427, 365: 428, 366: 429, 368: 430, 369: 431, 370: 432, 371: 433, 372: 434, 373: 435, 374: 436, 375: 437, 376: 438, 377: 439, 379: 440, 380: 441, 381: 442, 382: 443, 383: 444, 384: 445, 385: 446, 386: 447, 387: 448, 388: 449, 390: 450, 391: 451, 392: 452, 393: 453, 394: 454, 395: 455, 396: 456, 397: 457, 398: 458, 399: 459, 401: 460, 402: 461, 403: 462, 404: 463, 405: 464, 406: 465, 407: 466, 408: 467, 409: 468, 410: 469, 412: 470, 413: 471, 414: 472, 415: 473, 416: 474, 417: 475, 418: 476, 419: 477, 420: 478, 421: 479, 423: 480, 424: 481, 425: 482, 426: 483, 427: 484, 428: 485, 429: 486, 430: 487, 431: 488, 432: 489, 434: 490, 435: 491, 436: 492, 437: 493, 438: 494, 439: 495, 440: 496, 441: 497, 442: 498, 443: 499, 446: 500, 447: 501, 448: 502, 449: 503, 450: 504, 451: 505, 452: 506, 453: 507, 454: 508, 455: 509, 457: 510, 458: 511, 459: 512, 460: 513, 461: 514, 462: 515, 463: 516, 464: 517, 465: 518, 466: 519, 468: 520, 469: 521, 470: 522, 471: 523, 472: 524, 473: 525, 474: 526, 475: 527, 476: 528, 477: 529, 479: 530, 480: 531, 481: 532, 482: 533, 483: 534, 484: 535, 485: 536, 486: 537, 487: 538, 488: 539, 490: 540, 491: 541, 492: 542, 493: 543, 494: 544, 495: 545, 496: 546, 497: 547, 498: 548, 499: 549, 501: 550, 502: 551, 503: 552, 504: 553, 505: 554, 506: 555, 507: 556, 508: 557, 509: 558, 510: 559, 512: 560, 513: 561, 514: 562, 515: 563, 516: 564, 517: 565, 518: 566, 519: 567, 520: 568, 521: 569, 523: 570, 524: 571, 525: 572, 526: 573, 527: 574, 528: 575, 529: 576, 530: 577, 531: 578, 532: 579, 534: 580, 535: 581, 536: 582, 537: 583, 538: 584, 539: 585, 540: 586, 541: 587, 542: 588, 543: 589, 545: 590, 546: 591, 547: 592, 548: 593, 549: 594, 550: 595, 551: 596, 552: 597, 553: 598, 554: 599, 557: 600}

def validate_random_image(folder_path):
    """Validate a random image from the specified ImageLoader folder."""
    # now assume folder_path is the big folder containing the subfolders of the
    randomClass = random.randint(1, 600)
    classDir = os.path.join(folder_path, str(randomClass))
    image_files = [f for f in os.listdir(classDir)]
    selected_image = random.choice(image_files)
    image_path = os.path.join(folder_path, selected_image)
    image = Image.open(image_path)
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        pred_class = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][pred_class].item()

    print(f"Selected image: {selected_image}")
    print(f"Predicted Class: {class_map[pred_class]}")
    print(f"Confidence: {confidence:.5f}")
    
    if class_map[pred_class] == randomClass:
        print(f"Authentication successful!. Welcome {class_map[pred_class]}")
        # Add image to its corresponding folder
        # For simulation, this is just adding another copy of the image to the folder
        # Just in case of having similar names, we add a random number to the name
        new_image_name = f"{randomClass}_{random.randint(1, 10**6)}.PNG"
        new_image_path = os.path.join(folder_path, new_image_name)
        image.save(new_image_path)
    else:
        print(f"Authentication failed!")
        
    
    
# ==== Run it ====
if __name__ == "__main__":
    validate_random_image("dataloader/altered/train")