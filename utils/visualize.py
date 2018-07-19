import numpy as np
import cv2
import matplotlib.cm as cm
import PIL.Image as Image
import PIL.ImageFont as ImageFont
import PIL.ImageDraw as ImageDraw


def visualize_bboxes_on_image_v0(image, boxes, top_classes,
                              top_probs, class_labels):
    image_pil = Image.fromarray(image)
    for box, top_prob, top_class in zip(boxes, top_probs, top_classes):
        draw = ImageDraw.Draw(image_pil)
        ymin, xmin, ymax, xmax = box
        im_width, im_height = image_pil.size
        left, right, top, bottom = (xmin * im_width,
                                    xmax * im_width,
                                    ymin * im_height,
                                    ymax * im_height)
        top_pick = top_class[0]
        if top_class[0] == 0:
            top_pick = top_class[1]
        draw.line([(left, top), (left, bottom), (right, bottom),
                   (right, top), (left, top)], width=2, fill='red')
        draw.rectangle([(left, top), (left + 27, top + 20)], fill='red')
        font = ImageFont.truetype("Arial.ttf", 14)
        draw.text((left, top), class_labels[top_pick],
                  (255, 255, 255), font=font)
    np.copyto(image, np.array(image_pil))
    return image


def visualize_bboxes_on_image(image, boxes, top_classes,
                              top_probs, class_labels):
    im_height, im_width = image.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    labels_detected = []
    for box, top_prob, top_class in zip(boxes, top_probs, top_classes):
        ymin, xmin, ymax, xmax = box
        left, right, top, bottom = np.array([xmin * im_width,
                                    xmax * im_width,
                                    ymin * im_height,
                                    ymax * im_height]).astype(np.int16)
        if top_class[0] == 0:
            continue
        image = cv2.rectangle(image, (left, top), (right, bottom),
                              (255, 0, 0), 2)
        image = cv2.rectangle(image, (left, top), (left + 27, top + 20),
                              (255, 0, 0), -1)
        image = cv2.putText(image, str(top_class[0]), (left, top + 16),
                            font, 0.5, (255, 255, 255), 1)
        labels_detected.append(top_class[0])
        # class_labels[top_class[0]]
    labels_detected.sort()
    labels_detected = np.unique(labels_detected)
    legend = 255 * np.ones((320, im_width, 3), dtype=np.uint8)
    for i, l in enumerate(labels_detected):
        y = i * 20
        legend = cv2.rectangle(legend, (0, y), (im_width, y + 20),
                               (255, 0, 0), 2)
        legend = cv2.rectangle(legend, (0, y), (27, y + 20),
                               (255, 0, 0), -1)
        legend = cv2.putText(legend, str(l), (2, y + 16),
                             font, 0.5, (255, 255, 255), 1)
        legend = cv2.putText(legend, class_labels[l], (30, y + 16),
                             font, 0.5, (0, 0, 0), 1)
    image = np.concatenate([image, legend], axis=0)
    return image
