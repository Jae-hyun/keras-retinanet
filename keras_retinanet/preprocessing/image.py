import time
import numpy as np
import cv2

def random_transform_batch(
	image_batch,
	boxes_batch,
	image_data_generator,
	seed=None
):
	if seed is None:
		seed = np.uint32(time.time() * 1000)

	for batch in range(image_batch.shape[0]):
		image_batch[batch] = image_data_generator.random_transform(image_batch[0], seed=seed)

		# set fill mode so that masks are not enlarged
		fill_mode = image_data_generator.fill_mode
		image_data_generator.fill_mode = 'constant'

		for idx in range(boxes_batch.shape[1]):
			# generate box mask and randomly transform it
			mask = np.zeros_like(image_batch[batch], dtype=np.uint8)
			b = boxes_batch[batch, idx, :4].astype(int)
			cv2.rectangle(mask, (b[0], b[1]), (b[2], b[3]), (255,) * image_batch[batch].shape[-1], -1)
			mask = image_data_generator.random_transform(mask, seed=seed)[..., 0]
			mask = mask.copy() # to force contiguous arrays

			# find bounding box again in augmented image
			contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			contour = contours[0]
			x, y, w, h = cv2.boundingRect(contour)
			boxes_batch[batch, idx, 0] = x
			boxes_batch[batch, idx, 1] = y
			boxes_batch[batch, idx, 2] = x + w
			boxes_batch[batch, idx, 3] = y + h

		# restore fill_mode
		image_data_generator.fill_mode = fill_mode

	return image_batch, boxes_batch

