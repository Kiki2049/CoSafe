import os
import json
import random
import datetime
import pathlib
import time
import imp
from collections import deque
import math

import yaml
import cv2
import carla
import numpy as np
from PIL import Image

from leaderboard.autoagents import autonomous_agent
from team_code.lm_planner import RoutePlanner, InstructionPlanner
from team_code.lm_pid_controller import PIDController
from team_code.lmdrive_client import LMDriveClient
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

try:
	import pygame
except ImportError:
	raise RuntimeError("cannot import pygame, make sure pygame package is installed")


SAVE_PATH = os.environ.get("SAVE_PATH", 'eval')

def rotate_lidar(lidar, angle):
	radian = np.deg2rad(angle)
	return lidar @ [
		[ np.cos(radian), np.sin(radian), 0, 0],
		[-np.sin(radian), np.cos(radian), 0, 0],
		[0,0,1,0],
		[0,0,0,1]
	]

def lidar_to_raw_features(lidar):
	def preprocess(lidar_xyzr, lidar_painted=None):

		idx = (lidar_xyzr[:,0] > -1.2)&(lidar_xyzr[:,0] < 1.2)&(lidar_xyzr[:,1]>-1.2)&(lidar_xyzr[:,1]<1.2)

		idx = np.argwhere(idx)

		if lidar_painted is None:
			return np.delete(lidar_xyzr, idx, axis=0)
		else:
			return np.delete(lidar_xyzr, idx, axis=0), np.delete(lidar_painted, idx, axis=0)

	lidar_xyzr = preprocess(lidar)

	idxs = np.arange(len(lidar_xyzr))
	np.random.shuffle(idxs)
	lidar_xyzr = lidar_xyzr[idxs]

	lidar = np.zeros((40000, 4), dtype=np.float32)
	num_points = min(40000, len(lidar_xyzr))
	lidar[:num_points,:4] = lidar_xyzr
	lidar[np.isinf(lidar)] = 0
	lidar[np.isnan(lidar)] = 0
	lidar = rotate_lidar(lidar, -90).astype(np.float32)
	return lidar, num_points

class DisplayInterface(object):
	def __init__(self):
		self._width = 1200
		self._height = 900
		self._surface = None

		pygame.init()
		pygame.font.init()
		self._clock = pygame.time.Clock()
		self._display = pygame.display.set_mode(
			(self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF
		)
		pygame.display.set_caption("LMDrive Agent")

	def run_interface(self, input_data):
		rgb = input_data['rgb_front']
		rgb_left = input_data['rgb_left']
		rgb_right = input_data['rgb_right']
		rgb_focus = input_data['rgb_center']
		surface = np.zeros((900, 1200, 3),np.uint8)
		surface[:, :1200] = rgb
		surface[:210,:280] = input_data['rgb_left']
		surface[:210, 920:1200] = input_data['rgb_right']
		surface[:210, 495:705] = input_data['rgb_center']
		surface = cv2.putText(surface, input_data['time'], (20,710), cv2.FONT_HERSHEY_TRIPLEX,0.75,(0,0,255), 1)
		surface = cv2.putText(surface, input_data['meta_control'], (20,740), cv2.FONT_HERSHEY_TRIPLEX,0.75,(0,0,255), 1)
		surface = cv2.putText(surface, input_data['waypoints'], (20,770), cv2.FONT_HERSHEY_TRIPLEX,0.75,(0,0,255), 1)
		surface = cv2.putText(surface, input_data['instruction'], (20,800), cv2.FONT_HERSHEY_TRIPLEX,0.75,(0,0,255), 1)
		surface = cv2.putText(surface, input_data['notice'], (20,830), cv2.FONT_HERSHEY_TRIPLEX,0.75,(0,0,255), 1)

		surface = cv2.putText(surface, 'Left  View', (60,245), cv2.FONT_HERSHEY_TRIPLEX,0.75,(139,69,19), 2)
		surface = cv2.putText(surface, 'Focus View', (535,245), cv2.FONT_HERSHEY_TRIPLEX,0.75,(139,69,19), 2)
		surface = cv2.putText(surface, 'Right View', (980,245), cv2.FONT_HERSHEY_TRIPLEX,0.75,(139,69,19), 2)

		surface[:210, 278:282] = [139,69,19]
		surface[:210, 493:497] = [139,69,19]
		surface[:210, 703:707] = [139,69,19]
		surface[:210, 918:922] = [139,69,19]
		surface[208:212, :280] = [139,69,19]
		surface[208:212, 920:1200] = [139,69,19]
		surface[208:212, 495:705] = [139,69,19]

		self._surface = pygame.surfarray.make_surface(surface.swapaxes(0,1))

		if self._surface is not None:
			self._display.blit(self._surface, (0, 0))

		pygame.display.flip()
		pygame.event.get()
		return surface

	def _quit(self):
		pygame.quit()


def get_entry_point():
	return "LMDriveAgent"


class LMDriveAgent(autonomous_agent.AutonomousAgent):
	def setup(self, path_to_conf_file, ego_vehicles_num):
		self.agent_name = 'lmdrive'
		self._hic = DisplayInterface()
		self.track = autonomous_agent.Track.SENSORS
		self.step = -1
		self.wall_start = time.time()
		self.initialized = False

		self.active_misleading_instruction = [False] * 10
		self.remaining_misleading_frames = [0] * 10

		self.buffer_length = [0] * 10

		self.lm_config = imp.load_source("MainModel", path_to_conf_file).GlobalConfig()
		self.config = self.lm_config
		self.ego_vehicles_num = ego_vehicles_num #1
		self.turn_controller = [PIDController(K_P=self.lm_config.turn_KP, K_I=self.lm_config.turn_KI, K_D=self.lm_config.turn_KD, n=self.lm_config.turn_n) for _ in range(10)]
		self.speed_controller = [PIDController(K_P=self.lm_config.speed_KP, K_I=self.lm_config.speed_KI, K_D=self.lm_config.speed_KD, n=self.lm_config.speed_n) for _ in range(10)]

		self.agent_use_notice = self.lm_config.agent_use_notice
		self.traffic_light_notice = [''] * 10
		self.curr_notice = [''] * 10
		self.curr_notice_frame_id = [-1] * 10
		self.sample_rate = self.lm_config.sample_rate * 2 # The frequency of CARLA simulation is 20Hz

		session_id = os.environ.get("PORT", "default")
		self.lmdrive_client = LMDriveClient(
			os.environ.get("LMDRIVE_SERVER_URL", "http://localhost:8100"),
			session_id=session_id
		)

		self.prev_lidar = [None] * 10
		self.prev_control = [None] * 10
		self.curr_instruction = ['Drive Safely'] * 10
		self.sampled_scenarios = None
		self.instruction = ''

		self.save_path = None
		if SAVE_PATH is not None:
			now = datetime.datetime.now()
			string = pathlib.Path(os.environ["ROUTES"]).stem + "_"
			string += "_".join(
				map(
					lambda x: "%02d" % x,
					(now.month, now.day, now.hour, now.minute, now.second),
				)
			)

			print("Data save path:", string)

			self.save_path = pathlib.Path(SAVE_PATH) / string
			self.save_path.mkdir(parents=True, exist_ok=False)
			for ego_id in range(ego_vehicles_num):
				(self.save_path / f"meta_{ego_id}").mkdir(parents=True, exist_ok=False)

	def _init(self):
		self._route_planner = RoutePlanner(5, 50.0)
		self._route_planner.set_route(self._global_plan, True)
		self._instruction_planner = InstructionPlanner(notice_light_switch = True) # self.scenario_cofing_name,
		self.next_action_step = [-1] * 10
		# Reset server-side temporal buffers for all egos (new scenario)
		for eid in range(self.ego_vehicles_num):
			self.lmdrive_client.reset(eid)
			self.buffer_length[eid] = 0
		self.prev_lidar = [None] * 10
		self.initialized = True
		random.seed(''.join([str(x[0]) for x in self._global_plan]))
  
		self.debug_dict = {}
		for id in range(10):
			self.debug_dict[id] = {
				"num_misleading": 0,
				"num_instruction": 0,
    			"num_self.curr_instruction != last_instruction":0,
       			"num_last_misleading_instruction_exist": 0
			}


	def _get_position(self, tick_data):
		gps = tick_data["gps"]
		gps = (gps - self._route_planner.mean) * self._route_planner.scale
		return gps

	def sensors(self):
		return [
			{
				"type": "sensor.camera.rgb",
				"x": 1.3,
				"y": 0.0,
				"z": 2.3,
				"roll": 0.0,
				"pitch": 0.0,
				"yaw": 0.0,
				"width": 1200,
				"height": 900,
				"fov": 100,
				"id": "rgb_front",
			},
			{
				"type": "sensor.camera.rgb",
				"x": 1.3,
				"y": 0.0,
				"z": 2.3,
				"roll": 0.0,
				"pitch": 0.0,
				"yaw": -60.0,
				"width": 400,
				"height": 300,
				"fov": 100,
				"id": "rgb_left",
			},
			{
				"type": "sensor.camera.rgb",
				"x": 1.3,
				"y": 0.0,
				"z": 2.3,
				"roll": 0.0,
				"pitch": 0.0,
				"yaw": 60.0,
				"width": 400,
				"height": 300,
				"fov": 100,
				"id": "rgb_right",
			},
			{
				"type": "sensor.camera.rgb",
				"x": -1.3,
				"y": 0.0,
				"z": 2.3,
				"roll": 0.0,
				"pitch": 0.0,
				"yaw": 180.0,
				"width": 400,
				"height": 300,
				"fov": 100,
				"id": "rgb_rear",
			},
			{
				"type": "sensor.lidar.ray_cast",
				"x": 1.3,
				"y": 0.0,
				"z": 2.5,
				"roll": 0.0,
				"pitch": 0.0,
				"yaw": -90.0,
				"id": "lidar",
			},
			{
				"type": "sensor.other.imu",
				"x": 0.0,
				"y": 0.0,
				"z": 0.0,
				"roll": 0.0,
				"pitch": 0.0,
				"yaw": 0.0,
				"sensor_tick": 0.05,
				"id": "imu",
			},
			{
				"type": "sensor.other.gnss",
				"x": 0.0,
				"y": 0.0,
				"z": 0.0,
				"roll": 0.0,
				"pitch": 0.0,
				"yaw": 0.0,
				"sensor_tick": 0.01,
				"id": "gps",
			},
			{"type": "sensor.speedometer", "reading_frequency": 20, "id": "speed"},
		]

	def tick(self, ego_id, input_data):

		rgb_front = cv2.cvtColor(input_data[f"rgb_front_{ego_id}"][1][:, :, :3], cv2.COLOR_BGR2RGB)
		rgb_left = cv2.cvtColor(input_data[f"rgb_left_{ego_id}"][1][:, :, :3], cv2.COLOR_BGR2RGB)
		rgb_right = cv2.cvtColor(
			input_data[f"rgb_right_{ego_id}"][1][:, :, :3], cv2.COLOR_BGR2RGB
		)
		rgb_rear = cv2.cvtColor(
			input_data[f"rgb_rear_{ego_id}"][1][:, :, :3], cv2.COLOR_BGR2RGB
		)
		gps = input_data[f"gps_{ego_id}"][1][:2]
		speed = input_data[f"speed_{ego_id}"][1]["speed"]
		compass = input_data[f"imu_{ego_id}"][1][-1]
		if (
			math.isnan(compass) == True
		):  # It can happen that the compass sends nan for a few frames
			compass = 0.0

		result = {
			"rgb_front": rgb_front,
			"rgb_left": rgb_left,
			"rgb_right": rgb_right,
			'rgb_rear': rgb_rear,
			"gps": gps,
			"speed": speed,
			"compass": compass,
		}

		pos = self._get_position(result)

		lidar_data = input_data[f'lidar_{ego_id}'][1]
		result['raw_lidar'] = lidar_data

		lidar_unprocessed = lidar_data[..., :4]
		if self.prev_lidar[ego_id] is not None:
			lidar_unprocessed_full = np.concatenate([lidar_unprocessed, self.prev_lidar[ego_id]])
		else:
			lidar_unprocessed_full = lidar_unprocessed
		self.prev_lidar[ego_id] = lidar_unprocessed

		lidar_processed, num_points= lidar_to_raw_features(lidar_unprocessed_full)
		result['lidar'] = lidar_processed
		result['num_points'] = num_points

		result["gps"] = pos
		next_wp, next_cmd = self._route_planner.run_step(pos, ego_id)
		result["next_waypoint"] = next_wp
		result["next_command"] = next_cmd.value
		result['measurements'] = [pos[0], pos[1], compass, speed]
		result['speed'] = speed

		theta = compass + np.pi / 2
		R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

		local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
		local_command_point = R.T.dot(local_command_point)
		result["target_point"] = local_command_point

		return result


	def run_step(self, input_data, timestamp):
		control_all = []
		self.step += 1
		for ego_id in range(self.ego_vehicles_num):
			if CarlaDataProvider.get_hero_actor(hero_id=ego_id) is not None:
				control = self.run_step_single_vehicle(ego_id, input_data, timestamp)
				control_all.append(control)
			else:
				control_all.append(None)
		return control_all
	
	
	def run_step_single_vehicle(self, ego_id, input_data, timestamp):
		if not self.initialized:
			self._init()

		# self.step += 1
		
		tick_data = self.tick(ego_id, input_data)

		if self.step < 20:
			control = carla.VehicleControl()
			control.steer = float(0)
			control.throttle = float(0)
			control.brake = float(1)
			return control

		realtime_mode = os.environ.get('REALTIME_MODE', '0')
		if realtime_mode == '1':
			# REAL-TIME
			if self.step < self.next_action_step[ego_id] and self.step:
				return self.prev_control[ego_id]
		else:
			# NON-REAL-TIME
			if self.step % 5 != 0 and self.step:
				return self.prev_control[ego_id]

		
		start_time = time.time()
		velocity = tick_data["speed"]
		command = tick_data["next_command"]
		
		print(ego_id, command)

		# self.town_id = "Town07"
		last_instruction = self._instruction_planner.command2instruct(self.town_id, tick_data, self._route_planner.route, ego_id)
		if ego_id == 1:
			print(last_instruction)
		last_notice = self._instruction_planner.pos2notice(self.sampled_scenarios[0], tick_data)
		last_traffic_light_notice = self._instruction_planner.traffic_notice(tick_data, ego_id)
		last_misleading_instruction = self._instruction_planner.command2mislead(self.town_id, tick_data, ego_id)

		if last_notice == '':
			last_notice = last_traffic_light_notice

		self.debug_dict[ego_id]["num_last_misleading_instruction_exist"] += bool(last_misleading_instruction)          #TODO debug

		reset_buffer = False
		if self.curr_instruction[ego_id] != last_instruction or self.buffer_length[ego_id] > 400:
			#self.debug_dict[ego_id]["num_self.curr_instruction != last_instruction"] +=1             #TODO debug

			if self.remaining_misleading_frames[ego_id] > 0:
				self.remaining_misleading_frames[ego_id] = self.remaining_misleading_frames[ego_id] - 1
			else:
				self.active_misleading_instruction[ego_id] = False
				if last_misleading_instruction!= '' and random.random() < 0.2:
					self.curr_instruction[ego_id] = last_misleading_instruction
					self.active_misleading_instruction[ego_id] = True
					self.remaining_misleading_frames[ego_id] = 4
				else:
					self.curr_instruction[ego_id] = last_instruction
				reset_buffer = True
				self.buffer_length[ego_id] = 0
				self.curr_notice[ego_id] = ''
				self.curr_notice_frame_id[ego_id] = -1

		print(self.curr_instruction[ego_id])

		self.debug_dict[ego_id]["num_instruction"] +=1             #TODO debug

		if last_notice != '' and last_notice != self.curr_notice[ego_id]:
			self.curr_notice[ego_id] = last_notice
			self.curr_notice_frame_id[ego_id] = self.buffer_length[ego_id]

		use_notice = self.agent_use_notice

		# --- Remote inference ---
		waypoints_np, end_prob, buffer_length = self.lmdrive_client.infer(
			ego_id=ego_id,
			rgb_front=tick_data["rgb_front"],
			rgb_left=tick_data["rgb_left"],
			rgb_right=tick_data["rgb_right"],
			rgb_rear=tick_data["rgb_rear"],
			lidar=tick_data["lidar"],
			num_points=tick_data["num_points"],
			target_point=tick_data["target_point"],
			velocity=velocity,
			text_input=self.curr_instruction[ego_id],
			reset_buffer=reset_buffer,
			use_notice=use_notice,
			notice_text=self.curr_notice[ego_id],
			notice_frame_id=self.curr_notice_frame_id[ego_id],
		)

		if waypoints_np is None:
			# Fallback to previous control on timeout/error
			if self.prev_control[ego_id] is not None:
				return self.prev_control[ego_id]
			control = carla.VehicleControl()
			control.steer = 0.0
			control.throttle = 0.0
			control.brake = 1.0
			return control

		self.buffer_length[ego_id] = buffer_length

		steer, throttle, brake, metadata = self.control_pid(waypoints_np, velocity, ego_id)

		if end_prob > 0.75:
			self.lmdrive_client.reset(ego_id)
			self.buffer_length[ego_id] = 0
			self.curr_notice[ego_id] = ''
			self.curr_notice_frame_id[ego_id] = -1

		if brake < 0.05:
			brake = 0.0
		if brake > 0.1:
			throttle = 0.0

		control = carla.VehicleControl()
		control.steer = float(steer) * 0.8
		control.throttle = float(throttle)
		control.brake = float(brake)

		end_time = time.time()
		print(end_time-start_time)

		self.next_action_step[ego_id] = self.step + int((end_time - start_time) * 20)

		display_data = {}
		display_data['rgb_front'] = cv2.resize(tick_data['rgb_front'], (1200, 900))
		display_data['rgb_left'] = cv2.resize(tick_data['rgb_left'], (280, 210))
		display_data['rgb_right'] = cv2.resize(tick_data['rgb_right'], (280, 210))
		display_data['rgb_center'] = cv2.resize(tick_data['rgb_front'][330:570, 480:720], (210, 210))

		if self.active_misleading_instruction[ego_id]:
			self.debug_dict[ego_id]["num_misleading"] +=1             #TODO debug
			display_data['instruction'] = "Instruction: [Misleading] %s" % self.curr_instruction[ego_id]
		else:
			display_data['instruction'] = "Instruction: %s" % self.curr_instruction[ego_id]
		display_data['time'] = 'Time: %.3f. Frames: %d. End prob: %.2f' % (timestamp, self.buffer_length[ego_id], end_prob)
		display_data['meta_control'] = 'Throttle: %.2f. Steer: %.2f. Brake: %.2f' %(
			control.steer, control.throttle, control.brake
		)
		display_data['waypoints'] = 'Waypoints: (%.1f, %.1f), (%.1f, %.1f)' % (waypoints_np[0,0], -waypoints_np[0,1], waypoints_np[1,0], -waypoints_np[1,1])
		display_data['notice'] = "Notice: %s" % last_notice
		surface = self._hic.run_interface(display_data)
		tick_data['surface'] = surface
		self.prev_control[ego_id] = control
		if SAVE_PATH is not None:
			self.save(ego_id, tick_data)
		# print(self.debug_dict)
		return control

	def save(self, ego_id, tick_data):
		frame = (self.step - 20)
		Image.fromarray(tick_data["surface"]).save(
			self.save_path / f"meta_{ego_id}" / ("%04d.jpg" % frame)
		)
		return

	def destroy(self):
		pass

	def control_pid(self, waypoints, velocity, ego_id):
		'''
		Predicts vehicle control with a PID controller.
		Args:
			waypoints (np.ndarray): predicted waypoints (5, 2)
			velocity (float): speedometer input
		'''
		assert waypoints.shape == (5, 2)

		# flip y is (forward is negative in our waypoints)
		waypoints = waypoints.copy()
		waypoints[:,1] *= -1
		speed = velocity

		desired_speed = np.linalg.norm(waypoints[0] - waypoints[1]) * 2.0
		brake = desired_speed < self.lm_config.brake_speed or (speed / desired_speed) > self.lm_config.brake_ratio

		aim = (waypoints[1] + waypoints[0]) / 2.0
		angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
		if(speed < 0.01):
			angle = np.array(0.0) # When we don't move we don't want the angle error to accumulate in the integral
		steer = self.turn_controller[ego_id].step(angle)
		steer = np.clip(steer, -1.0, 1.0)

		delta = np.clip(desired_speed - speed, 0.0, self.lm_config.clip_delta)
		throttle = self.speed_controller[ego_id].step(delta)
		throttle = np.clip(throttle, 0.0, self.lm_config.max_throttle)
		throttle = throttle if not brake else 0.0

		metadata = {
			'speed': float(speed.astype(np.float64)),
			'steer': float(steer),
			'throttle': float(throttle),
			'brake': float(brake),
			'wp_2': tuple(waypoints[1].astype(np.float64)),
			'wp_1': tuple(waypoints[0].astype(np.float64)),
			'desired_speed': float(desired_speed.astype(np.float64)),
			'angle': float(angle.astype(np.float64)),
			'aim': tuple(aim.astype(np.float64)),
			'delta': float(delta.astype(np.float64)),
		}

		return steer, throttle, brake, metadata
