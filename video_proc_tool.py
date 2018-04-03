import cv2
import os
import ConfigParser
import time
import sys

objdet_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'deep_objdetect')
if objdet_path not in sys.path:
	sys.path.insert(0, objdet_path)

import deep_objdetect.objdet_experiments as objexp

class misc_video_params():
	def __init__(self, config):
		self.input_directory = config.get('data', 'input_directory')
		self.output_directory = config.get('data', 'output_directory')

class objdet_video:
	def __init__(self, config_filename):
		self.config = ConfigParser.ConfigParser()
		self.config.read(config_filename)

		self.misc = misc_video_params(self.config)

		self.use_video_gen = self.config.getboolean('VideoObjectDetection', 'use_video_gen')

		if self.use_video_gen:
			self.fps = 1.0/float(self.config.get('VideoObjectDetection', 'interval'))

			self.deep_objdetect_config = self.config.get('VideoObjectDetection', 'deep_objdetect_config')
			self.objdet_alg = self.config.get('VideoObjectDetection', 'objdet_alg')
			if self.objdet_alg == 'faster_rcnn':
				self.deep_detector = objexp.faster_rcnn_module(self.deep_objdetect_config)
				self.deep_detector.faster_rcnn_online_init()
			elif self.objdet_alg == 'yolo':
				self.deep_detector = objexp.yolo_module(self.deep_objdetect_config)
				self.deep_detector.yolo_online_init()			
			else:
				assert False, 'INVALID DEEP OBJ DETECTION ALGORITHM GIVEN'

	def displayDets(self, image, all_dets):
		for det in all_dets:
			name = det[0]
			score = det[1]
			bbox = det[2]
			cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(0,255,0),2)
			cv2.putText(image, name, (int(bbox[0]),int(bbox[1])), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,255,255), 1)
			#print 'Class Name: {:s}, Score: {:f}, Bounding Box: [x1(left): {:d}, y1(top): {:d}, x2(right): {:d}, y2(bottom): {:d} ]'.format(name, score, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
		# cv2.imshow('display', image)
		# cv2.waitKey(0)
		return image

	def get_output_video(self, input_video_path, output_video_path):
		print '--------------'
		print 'Processing Video: ', input_video_path
		print 'Storing Result Images in: ', output_video_path
		print '--------------'
		
		inputvid = cv2.VideoCapture(input_video_path)

		if (not inputvid.isOpened):
			print 'ERROR OPENING VIDEO FILE: {:s}'.format(input_video_path)
			return

		videoFPS = float(inputvid.get(5))
		if videoFPS <= 0:
			print 'ERROR WITH FPS OF VIDEO FILE: {:s}'.format(input_video_path)
			return

		if (self.fps == -1) or (self.fps > videoFPS):
			self.fps = videoFPS

		frameInterval = int(round(float(videoFPS)/float(self.fps)))
		frameCount = 0

		codec = cv2.VideoWriter_fourcc('X','2','6','4')
		vidfps = self.fps
		frame_width = int(inputvid.get(3))
		frame_height = int(inputvid.get(4))
		outputvid = cv2.VideoWriter(output_video_path, codec, vidfps, (frame_width, frame_height))

		while (inputvid.isOpened()):
			ret, frame = inputvid.read()
			if ret:
				if (frameCount % frameInterval == 0):
					if self.objdet_alg == 'faster_rcnn':
						all_dets = self.deep_detector.faster_rcnn_online(frame)
					elif self.objdet_alg == 'yolo':
						all_dets = self.deep_detector.yolo_online(frame)

					out_frame = self.displayDets(frame, all_dets)
					outputvid.write(out_frame)

				frameCount = frameCount + 1
			else:
				break

		inputvid.release()
		outputvid.release()

		return

	def generate_videos(self):
		if self.use_video_gen:
			for dirpath, dirnames, filenames in os.walk(self.misc.input_directory):
				out_dir = os.path.join(self.misc.output_directory, dirpath[len(self.misc.input_directory)+1:])
				if not os.path.isdir(out_dir):
					os.makedirs(out_dir)

				for filename in filenames:
					input_video_path = os.path.join(dirpath, filename)
					output_video_path = os.path.join(out_dir, filename)
					self.get_output_video(input_video_path, output_video_path)
		return


class extract_images_from_video:
	def __init__(self, config_filename):
		self.config = ConfigParser.ConfigParser()
		self.config.read(config_filename)

		self.misc = misc_video_params(self.config)

		self.use_extract_image_video = self.config.getboolean('ExtractImagesVideo','use_extract_image_video')

		if self.use_extract_image_video:
			self.fps = 1.0/float(self.config.get('ExtractImagesVideo', 'interval'))
			self.image_format = self.config.get('ExtractImagesVideo', 'image_format')
			self.use_objdet = self.config.getboolean('ExtractImagesVideo', 'use_objdet')
			if self.use_objdet:
				self.deep_objdetect_config = self.config.get('ExtractImagesVideo', 'deep_objdetect_config')
				self.objdet_alg = self.config.get('ExtractImagesVideo', 'objdet_alg')
				self.num_objects_thresh = int(self.config.get('ExtractImagesVideo','num_objects_thresh'))
				if self.objdet_alg == 'faster_rcnn':
					self.deep_detector = objexp.faster_rcnn_module(self.deep_objdetect_config)
					self.deep_detector.faster_rcnn_online_init()
				elif self.objdet_alg == 'yolo':
					self.deep_detector = objexp.yolo_module(self.deep_objdetect_config)
					self.deep_detector.yolo_online_init()
				else:
					assert False, 'INVALID DEEP OBJ DETECTION ALGORITHM GIVEN'

	def store_images_from_video(self, input_video_path, output_folder):
		if not os.path.isdir(output_folder):
			os.makedirs(output_folder)
		
		print '--------------'
		print 'Processing Video: ', input_video_path
		print 'Storing Result Images in: ', output_folder
		print '--------------'
		
		inputvid = cv2.VideoCapture(input_video_path)

		if (not inputvid.isOpened):
			print 'ERROR OPENING VIDEO FILE: {:s}'.format(input_video_path)
			return

		videoFPS = float(inputvid.get(5))
		if videoFPS <= 0:
			print 'ERROR WITH FPS OF VIDEO FILE: {:s}'.format(input_video_path)
			return

		if (self.fps == -1) or (self.fps > videoFPS):
			self.fps = videoFPS

		frameInterval = int(round(float(videoFPS)/float(self.fps)))

		seqid = 0
		frameCount = 0

		while (inputvid.isOpened()):
			ret, frame = inputvid.read()
			if ret:
				if (frameCount % frameInterval == 0):
					if self.use_objdet:
						if self.objdet_alg == 'faster_rcnn':
							all_dets = self.deep_detector.faster_rcnn_online(frame)
						elif self.objdet_alg == 'yolo':
							all_dets = self.deep_detector.yolo_online(frame)

						num_dets = len(all_dets)

						if num_dets >= self.num_objects_thresh:
							out_image_file = os.path.join(output_folder, '{:010d}'.format(seqid) + self.image_format)
							print out_image_file
							cv2.imwrite(out_image_file, frame)
							seqid = seqid + 1
						else:
							print 'Not enough detections!!'
					else:
						out_image_file = os.path.join(output_folder, '{:010d}'.format(seqid) + self.image_format)
						print out_image_file
						cv2.imwrite(out_image_file, frame)
						seqid = seqid + 1

				frameCount = frameCount + 1
			else:
				break
		
		inputvid.release()

		return

	def extract_images(self):
		if self.use_extract_image_video:
			for dirpath, dirnames, filenames in os.walk(self.misc.input_directory):
				out_dir = os.path.join(self.misc.output_directory, dirpath[len(self.misc.input_directory)+1:])

				for filename in filenames:
					input_video_path = os.path.join(dirpath, filename)
					output_folder = os.path.splitext(os.path.join(out_dir, filename))[0]
					self.store_images_from_video(input_video_path, output_folder)
		return

if __name__ == '__main__':
	extfromvid = extract_images_from_video('video_proc_params.ini')
	extfromvid.extract_images()

	objdetvid = objdet_video('video_proc_params.ini')
	objdetvid.generate_videos()