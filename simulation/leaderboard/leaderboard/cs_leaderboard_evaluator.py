"""CoSafe Leaderboard Evaluator.

Inherits from LeaderboardEvaluator (leaderboard_evaluator_parameter.py) and
overrides only what's needed to inject CoSafe middleware:
- Replaces ScenarioManager with CsScenarioManager
- Spawns CoSafe sensors (RSU + vehicle cameras)
- Creates CoSafeRSU module and wires it into the scenario manager
"""

from __future__ import print_function

import traceback
import argparse
from argparse import RawTextHelpFormatter
import os
import sys
import json
import yaml
import numpy as np
import random
import torch

import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.leaderboard_evaluator_parameter import (
    LeaderboardEvaluator,
    Logger,
    backup_script,
    sensors_to_icons,
    check_log_file,
)
from leaderboard.cs_scenario_manager import CsScenarioManager
from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.envs.sensor_interface import SensorConfigurationInvalid
from leaderboard.autoagents.agent_wrapper import AgentWrapper, AgentError
from leaderboard.utils.statistics_manager import StatisticsManager
from leaderboard.utils.route_indexer import RouteIndexer

from cs_cosafe_c.cs_config import CsCoSafeConfig
from cs_cosafe_c.cs_rsu_sensor import CsSensorManager
from cs_cosafe_c.cs_rsu_module import CoSafeRSU


class CsLeaderboardEvaluator(LeaderboardEvaluator):
    """LeaderboardEvaluator with CoSafe safety middleware integration."""

    def __init__(self, args, statistics_manager):
        super().__init__(args, statistics_manager)
        self.manager = CsScenarioManager(args.timeout, args.debug > 1)
        self._cosafe_config = None
        self._cosafe_sensor_manager = None
        self._cosafe_rsu = None

    def _cleanup(self):
        """Cleanup CoSafe components before standard cleanup."""
        if self._cosafe_sensor_manager:
            self._cosafe_sensor_manager.cleanup()
            self._cosafe_sensor_manager = None
        if self._cosafe_rsu:
            self._cosafe_rsu.destroy()
            self._cosafe_rsu = None
        super()._cleanup()

    def _load_and_run_scenario(self, args, config):
        """Load and run scenario with CoSafe middleware injection.

        Same as parent but after world load + ego prepare + RouteScenario
        creation, spawns CoSafe sensors and creates CoSafeRSU before
        manager.load_scenario().
        """
        crash_message = ""
        entry_status = "Started"

        print("\n\033[1m========= Preparing {} (repetition {}) =========".format(
            config.name, config.repetition_index))
        print("> Setting up the agent\033[0m")

        for j in range(self.ego_vehicles_num):
            self.statistics_manager[j].set_route(config.name, config.index)

        try:
            self._agent_watchdog.start()
            agent_class_name = getattr(self.module_agent, 'get_entry_point')()
            self.agent_instance = getattr(self.module_agent, agent_class_name)(
                args.agent_config, self.ego_vehicles_num)
            config.agent = self.agent_instance
            if hasattr(self.agent_instance, "get_save_path"):
                print("Data Generation Confirmed!")
                config.save_path_root = self.agent_instance.get_save_path()
                log_root_dir = config.save_path_root
            else:
                print("Evaluation Process!")
                config.save_path_root = None
                log_root_dir = os.path.dirname(os.environ["CHECKPOINT_ENDPOINT"])
                try:
                    log_root_dir = self.agent_instance.get_save_path()
                except:
                    print('load save path failed')

            log_dir = os.path.join(log_root_dir, 'log')
            self.log_dir = log_dir
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            log_file_dir = os.path.join(log_dir, 'log.log')
            self.log_file_dir = log_file_dir
            sys.stdout = Logger(log_file_dir)
            args_file_dir = os.path.join(log_dir, 'args.json')
            args_dict = vars(args)
            json_str = json.dumps(args_dict, indent=2)
            with open(args_file_dir, 'w') as json_file:
                json_file.write(json_str)
            print("{} (repetition {}) Log file initialized!".format(
                config.name, config.repetition_index))

            backup_script(os.environ["RESULT_ROOT"])

            if not self.sensors:
                self.sensors = self.agent_instance.sensors()
                track = self.agent_instance.track
                AgentWrapper.validate_sensor_configuration(self.sensors, track, args.track)
                self.sensor_icons = [sensors_to_icons[sensor['type']] for sensor in self.sensors]
                for j in range(self.ego_vehicles_num):
                    self.statistics_manager[j].save_sensors(self.sensor_icons, args.checkpoint)
            self._agent_watchdog.stop()

        except SensorConfigurationInvalid as e:
            print("\n\033[91mThe sensor's configuration used is invalid:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()
            traceback.print_exc(file=open(log_file_dir, 'a'))
            crash_message = "Agent's sensors were invalid"
            entry_status = "Rejected"
            self._cleanup()
            sys.exit(-1)

        except Exception as e:
            print("\n\033[91mCould not set up the required agent:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()
            traceback.print_exc(file=open(log_file_dir, 'a'))
            crash_message = "Agent couldn't be set up"
            self._cleanup()
            return

        print("\033[1m> Loading the world\033[0m")
        try:
            self._load_and_wait_for_world(args, config.town, config.ego_vehicles)
            self._prepare_ego_vehicles(config.ego_vehicles, False)
            with open(args.scenario_parameter, 'r', encoding='utf-8') as f:
                scenario_parameter = yaml.load(f.read(), Loader=yaml.FullLoader)
            if scenario_parameter['Background']['turn_off_light']:
                print("crazy traffic lights")
                [tf.set_state(carla.libcarla.TrafficLightState.Green)
                 for tf in self.world.get_actors().filter("*traffic_light*")
                 if hasattr(tf, "set_state")]
                [tf.freeze(True)
                 for tf in self.world.get_actors().filter("*traffic_light*")
                 if hasattr(tf, "freeze")]

            scenario = RouteScenario(
                world=self.world, config=config, debug_mode=args.debug,
                ego_vehicles_num=self.ego_vehicles_num, log_dir=log_dir,
                scenario_parameter=scenario_parameter)
            config.trajectory = scenario.get_new_config_trajectory()
            if self.ego_vehicles_num != 1:
                for j in range(self.ego_vehicles_num):
                    self.statistics_manager[j].set_scenario(scenario.scenario[j])
            else:
                self.statistics_manager[0].set_scenario(scenario.scenario)

            if config.weather.sun_altitude_angle < 0.0:
                for vehicle in scenario.ego_vehicles:
                    vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))

            if args.record:
                self.client.start_recorder("{}/{}_rep{}.log".format(
                    args.record, config.name, config.repetition_index))
            self.manager.load_scenario(
                scenario, self.agent_instance, config.repetition_index,
                self.ego_vehicles_num, save_root=config.save_path_root,
                sensor_tf_list=scenario.get_sensor_tf(),
                is_crazy=scenario_parameter['Background']['turn_off_light'])

            print("\033[1m> [CoSafe] Setting up safety middleware\033[0m")
            self._cosafe_config = CsCoSafeConfig()
            ablation = os.environ.get('COSAFE_ABLATION', '').strip()
            if ablation == 'no_coop':
                self._cosafe_config.perception.cooperative_perception = False
            elif ablation == 'no_risk':
                self._cosafe_config.risk.enable_deadlock = False
            elif ablation == 'no_joint':
                self._cosafe_config.selector.selector_mode = 'independent'
            elif ablation:
                print("[CoSafe] WARN: unknown COSAFE_ABLATION={!r} — ignoring".format(ablation))
            if ablation:
                print("[CoSafe] Ablation mode: {}".format(ablation))
            self._cosafe_sensor_manager = CsSensorManager(self._cosafe_config)
            self._cosafe_sensor_manager.spawn_rsu(scenario.ego_vehicles)
            self._cosafe_sensor_manager.spawn_vehicle_sensors(scenario.ego_vehicles)
            self._cosafe_rsu = CoSafeRSU(self._cosafe_config, self.ego_vehicles_num)
            cosafe_save_root = (config.save_path_root
                                or os.environ.get('SAVE_PATH'))
            self._cosafe_rsu.setup(
                ego_vehicles=scenario.ego_vehicles,
                rsu_location=self._cosafe_sensor_manager.rsu_location,
                save_root=cosafe_save_root,
            )

            self.manager.set_cosafe_rsu(self._cosafe_rsu)
            self.manager.set_cosafe_sensor_manager(self._cosafe_sensor_manager)

        except Exception as e:
            print("\n\033[91mThe scenario could not be loaded:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()
            traceback.print_exc(file=open(log_file_dir, 'a'))

            crash_message = "Simulation crashed"
            entry_status = "Crashed"

            self._register_statistics(config, args.ego_num, args.checkpoint,
                                      entry_status, crash_message)

            if args.record:
                self.client.stop_recorder()

            self._cleanup()
            sys.exit(-1)

        print("\033[1m> Running the route\033[0m")

        try:
            self.manager.run_scenario()
        except AgentError as e:
            print("\n\033[91mStopping the route, the agent has crashed:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()
            traceback.print_exc(file=open(log_file_dir, 'a'))
            crash_message = "Agent crashed"
        except Exception as e:
            print("\n\033[91mError during the simulation:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()
            traceback.print_exc(file=open(log_file_dir, 'a'))
            crash_message = "Run scenario crashed"
            entry_status = "Crashed"

        try:
            print("\033[1m> Stopping the route\033[0m")
            self.manager.stop_scenario()
            self._register_statistics(config, args.ego_num, args.checkpoint,
                                      entry_status, crash_message)

            if args.record:
                self.client.stop_recorder()

            scenario.remove_all_actors()

            self._cleanup()
            for zombie in self.world.get_actors().filter("*sensor*"):
                zombie.stop()
                zombie.destroy()
                zombie = None

        except Exception as e:
            print("\n\033[91mFailed to stop the scenario, the statistics might be empty:")
            print("> {}\033[0m\n".format(e))
            traceback.print_exc()
            traceback.print_exc(file=open(log_file_dir, 'a'))
            crash_message = "Simulation crashed"


def main():
    description = "CoSafe CARLA AD Leaderboard Evaluation: evaluate your Agent with CoSafe safety middleware\n"

    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default='50010',
                        help='TCP port to listen to (default: 2000)')
    parser.add_argument('--trafficManagerPort', default='50050',
                        help='Port to use for the TrafficManager (default: 8000)')
    parser.add_argument('--trafficManagerSeed', default='1',
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--carlaProviderSeed', default='2000',
                        help='Seed used by the CarlaProvider (default: 2000)')
    parser.add_argument('--debug', type=int, help='Run with debug output', default=0)
    parser.add_argument('--record', type=str, default='',
                        help='Use CARLA recording feature to create a recording of the scenario')
    parser.add_argument('--timeout', default="600.0",
                        help='Set the CARLA client timeout value in seconds')

    parser.add_argument('--routes', default=None,
                        help='Name of the route to be executed.')
    parser.add_argument('--routes_0', default=None,
                        help='Name of the route to be executed by ego vehicle 0.')
    parser.add_argument('--routes_1', default=None,
                        help='Name of the route to be executed by ego vehicle 1.')
    parser.add_argument('--routes_2', default=None,
                        help='Name of the route to be executed by ego vehicle 2.')
    parser.add_argument('--routes_dir', default=None,
                        help='the directory that contains all routes for multiple vehicles')
    parser.add_argument('--scenarios',
                        default='simulation/leaderboard/data/scenarios/town05_all_scenarios_2.json',
                        help='Name of the scenario annotation file to be mixed with the route.')
    parser.add_argument('--scenario_parameter',
                        default='simulation/leaderboard/leaderboard/scenarios/scenario_parameter_demo_0.yaml',
                        help='Defination of the scenario parameters.')
    parser.add_argument('--repetitions', type=int, default=1,
                        help='Number of repetitions per route.')

    parser.add_argument("-a", "--agent", type=str,
                        default='simulation/leaderboard/team_code/pnp_agent_e2e_demo.py',
                        help="Path to Agent's py file to evaluate")
    parser.add_argument("--agent-config", type=str,
                        default='simulation/leaderboard/team_code/agent_config_e2e/pnp_config_codriving_5_10.yaml',
                        help="Path to Agent's configuration file")

    parser.add_argument("--track", type=str, default='SENSORS',
                        help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=int, default=0,
                        help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str,
                        default='/GPFS/data/gjliu-1/Auto-driving/V2Xverse/out_dir/demo/results.json',
                        help="Path to checkpoint used for saving statistics and resuming")
    parser.add_argument('--ego-num', type=int, default=1,
                        help='The number of ego vehicles')
    parser.add_argument('--skip_existed', type=int, default=1,
                        help='If the result exist, return')

    arguments = parser.parse_args()
    print("start (CoSafe evaluator)")
    check_result = False

    if arguments.skip_existed:
        if os.path.exists(os.path.join(os.path.dirname(arguments.checkpoint),
                                       "ego_vehicle_{}".format(0),
                                       os.path.basename(arguments.checkpoint))):
            image_save_path = os.environ['SAVE_PATH']
            route_list = os.listdir(image_save_path)
            route_list.sort()
            log_dir = route_list[-1]
            log_file_path = "{}/{}/log/log.log".format(image_save_path, log_dir)

            for invalid_str in ['Traceback', 'No space']:
                if check_log_file(log_file_path, invalid_str):
                    check_result = True
                    break

            if not check_log_file(log_file_path, 'RouteCompletionTest'):
                check_result = True

            if check_result:
                print('Invalid results, rerun!')
            else:
                print('Valid results, skip!')
                return
        else:
            print('{} do not exists, continue!'.format(
                os.path.join(os.path.dirname(arguments.checkpoint),
                             "ego_vehicle_{}".format(0),
                             os.path.basename(arguments.checkpoint))))

    if not os.path.exists(os.environ["SAVE_PATH"]):
        os.makedirs(os.environ["SAVE_PATH"])
    if not os.path.exists(os.path.dirname(os.environ["CHECKPOINT_ENDPOINT"])):
        os.makedirs(os.path.dirname(os.environ["CHECKPOINT_ENDPOINT"]))

    if "RESULT_ROOT" not in os.environ:
        os.environ["RESULT_ROOT"] = os.environ["DATA_ROOT"]

    statistics_manager_all = []
    for i in range(arguments.ego_num):
        statistics_manager = StatisticsManager(ego_car_id=i)
        statistics_manager_all.append(statistics_manager)

    try:
        leaderboard_evaluator = CsLeaderboardEvaluator(arguments, statistics_manager_all)
        leaderboard_evaluator.run(arguments)

    except Exception as e:
        traceback.print_exc()
    finally:
        del leaderboard_evaluator


if __name__ == '__main__':
    main()
