import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import Any, Callable, Dict, List

from mbodied.agents.language import LanguageAgent
from mbodied.agents.motion.openvla_agent import OpenVlaAgent
from mbodied.agents.sense.audio.audio_agent import AudioAgent
from mbodied.agents.sense.object_pose_estimator_3d import ObjectPoseEstimator3D
from mbodied.hardware.sim_interface import SimInterface
from mbodied.types.message import Message
from mbodied.types.sense.vision import Image


@dataclass
class AgentTask:
    name: str
    agent: str
    inputs: List[str]
    outputs: List[str]
    is_queue: bool
    func: Callable


class HRI:
    """Human-Robot Interaction (HRI) class to coordinate the interaction between the robot and the user."""

    def __init__(self, agents: Dict[str, Any], tasks: List[AgentTask]) -> None:
        self.agents = agents
        self.tasks = {task.name: task for task in tasks}
        self.states = self.initialize_states()

        self.stop_event = threading.Event()
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        self.threads = {
            name: threading.Thread(target=self.create_thread(task.agent, task.inputs, task.outputs, task.func))
            for name, task in self.tasks.items()
        }

    def initialize_states(self) -> Dict[str, Any]:
        states = {}
        for task in self.tasks.values():
            for output in task.outputs:
                if output not in states:
                    states[output] = Queue() if task.is_queue else None
            for input_ in task.inputs:
                if input_ not in states:
                    # TODO: is there concurrency issue?
                    states[input_] = None
        return states

    def create_thread(self, agent, inputs, outputs, func) -> Callable:
        def thread_func():
            while not self.stop_event.is_set():
                try:
                    input_values = []
                    all_available = True

                    # Check availability of all input values
                    for input_ in inputs:
                        if isinstance(self.states[input_], Queue):
                            if self.states[input_].empty():
                                all_available = False
                                break
                            input_values.append(self.states[input_].get())
                        else:
                            if self.states[input_] is None:
                                all_available = False
                                break
                            input_values.append(self.states[input_])

                    # If not all inputs are available, wait and continue
                    if not all_available:
                        time.sleep(0.1)
                        continue

                    # Process the inputs
                    output_value = func(self.agents[agent], *input_values)
                    if not isinstance(output_value, tuple):
                        output_value = (output_value,)
                    for output, value in zip(outputs, output_value):
                        if isinstance(self.states[output], Queue):
                            self.states[output].put(value)
                        else:
                            self.states[output] = value
                except Exception as e:
                    logging.error(f"{agent} thread error: {e}", exc_info=True)
                    # TODO: remove this after debug
                    exit()

        return thread_func

    def run(self) -> None:
        for thread in self.threads.values():
            thread.start()
        logging.info("All threads started.")

    def stop(self) -> None:
        self.stop_event.set()
        for thread in self.threads.values():
            thread.join()
        logging.info("All threads stopped.")

    def signal_handler(self, sig, frame) -> None:
        logging.info("Signal received, exiting program...")
        sys.exit(0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    agents = {
        "pose_agent": ObjectPoseEstimator3D(),
        "audio_agent": AudioAgent(),
        "language_agent": LanguageAgent(model_src="openai"),
        "motor_agent": OpenVlaAgent(model_src="https://api.mbodi.ai/community-models/"),
        "robot": SimInterface(),
    }

    tasks = [
        AgentTask(name="capture_image", agent="robot", inputs=[], outputs=["observation"], is_queue=False, func=lambda robot: robot.capture()),
        # AgentTask(name="pose_estimation", agent="pose_agent", inputs=["observation"], outputs=["object_pose"], is_queue=False, func=lambda agent, obs: agent.act(obs)),
        AgentTask(name="audio_task", agent="audio_agent", inputs=[], outputs=["instruction"], is_queue=False, func=lambda agent: agent.act()),
        AgentTask(name="language_task", agent="language_agent", inputs=["instruction"], outputs=["language_instruction"], is_queue=False, func=lambda agent, instr: agent.act(instr)),
        AgentTask(name="motor_task", agent="motor_agent", inputs=["language_instruction", "observation"], outputs=["motion"], is_queue=True, func=lambda agent, lang_instr, obs: agent.act(lang_instr, obs)),
        AgentTask(name="robot_task", agent="robot", inputs=["motion", "observation"], outputs=[], is_queue=False, func=lambda robot, motion: robot.do(motion)),
    ]

    hri = HRI(agents, tasks)
    hri.run()
