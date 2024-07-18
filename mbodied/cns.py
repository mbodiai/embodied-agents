import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import Any, Callable, Dict, List


@dataclass
class AgentTask:
    """Class representing a task to be executed by an agent."""

    name: str
    agent: str
    inputs: List[str]
    outputs: List[str]
    is_queue: bool
    func: Callable


class CentralNervousSystem:
    """CentralNervousSystem coordinates the interaction between different agents within the robot.

    Every task is associated with an agent and has a set of inputs and outputs.

    This class is responsible for:
    - Initializing and managing the states for inputs and outputs of tasks.
    - Creating and managing threads for each task to be executed concurrently.
    - Handling signals for graceful shutdown.
    - Coordinating the flow of data between agents and their tasks.

    Attributes:
        agents (Dict[str, Any]): Dictionary mapping agent names to agent instances.
        tasks (Dict[str, AgentTask]): Dictionary mapping task names to AgentTask instances.
        states (Dict[str, Any]): Dictionary holding the states for inputs and outputs.
        stop_event (threading.Event): Event to signal threads to stop.
        threads (Dict[str, threading.Thread]): Dictionary mapping task names to their corresponding threads.
    """

    def __init__(self, agents: Dict[str, Any], tasks: List[AgentTask]) -> None:
        """Initialize the CentralNervousSystem.

        Args:
            agents (Dict[str, Any]): A dictionary of agent names and their instances.
            tasks (List[AgentTask]): A list of AgentTask instances.
        """
        self.agents = agents
        self.tasks = {task.name: task for task in tasks}
        self.states, self.state_locks = self.initialize_states()

        self.stop_event = threading.Event()
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        self.threads = {
            name: threading.Thread(target=self.create_thread(task.agent, task.inputs, task.outputs, task.func))
            for name, task in self.tasks.items()
        }

    def initialize_states(self) -> (Dict[str, Any], Dict[str, threading.Lock]):
        """Initialize the states for inputs and outputs.

        Returns:
            Tuple[Dict[str, Any], Dict[str, threading.Lock]]: A dictionary mapping state names to their initial values and their locks.
        """
        states = {}
        state_locks = {}
        for task in self.tasks.values():
            for output in task.outputs:
                if output not in states:
                    states[output] = Queue() if task.is_queue else None
                    state_locks[output] = threading.Lock()
            for input_ in task.inputs:
                if input_ not in states:
                    states[input_] = None
                    state_locks[input_] = threading.Lock()
        return states, state_locks

    def create_thread(self, agent, inputs, outputs, func) -> Callable:
        """Create a function to be run in a thread for a given task.

        Args:
            agent (str): The name of the agent responsible for this task.
            inputs (List[str]): List of input state names.
            outputs (List[str]): List of output state names.
            func (Callable): The function to be executed by the agent.

        Returns:
            Callable: A function to be run in a thread.
        """

        def thread_func():
            while not self.stop_event.is_set():
                try:
                    input_values = []
                    all_available = True

                    # Check availability of all input values
                    for input_ in inputs:
                        with self.state_locks[input_]:
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
                                self.states[input_] = None

                    # If not all inputs are available, wait and continue
                    if not all_available:
                        time.sleep(0.1)
                        continue

                    # Execute the task function with input values and update the output states
                    output_value = func(self.agents[agent], *input_values)
                    if not isinstance(output_value, tuple):
                        output_value = (output_value,)

                    for output, value in zip(outputs, output_value):
                        with self.state_locks[output]:
                            if isinstance(self.states[output], Queue):
                                self.states[output].put(value)
                            else:
                                self.states[output] = value

                except Exception as e:
                    logging.error(f"{agent} thread error: {e}", exc_info=True)
                    self.stop_event.set()

        return thread_func

    def run(self) -> None:
        """Start all threads and keep the main program running."""
        for thread in self.threads.values():
            thread.start()
        logging.info("All threads started.")

        try:
            while not self.stop_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received, stopping...")
        finally:
            self.stop()

    def stop(self) -> None:
        """Signal all threads to stop and wait for them to finish."""
        self.stop_event.set()
        for thread in self.threads.values():
            thread.join()
        logging.info("All threads stopped.")

    def signal_handler(self, sig, frame) -> None:
        """Handle incoming signals and exit the program gracefully."""
        logging.info("Signal received, exiting program...")
        sys.exit(0)
