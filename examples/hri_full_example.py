import logging

from mbodied.agents.language import LanguageAgent
from mbodied.agents.motion.openvla_agent import OpenVlaAgent
from mbodied.agents.sense.audio.audio_agent import AudioAgent
from mbodied.agents.sense.object_pose_estimator_3d import ObjectPoseEstimator3D
from mbodied.hardware.sim_interface import SimInterface
from mbodied.hri.hri import HRI, AgentTask
from mbodied.types.motion.control import HandControl

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
        AgentTask(name="pose_estimation", agent="pose_agent", inputs=["observation"], outputs=["object_pose"], is_queue=False, func=lambda agent, obs: agent.act(obs)),
        AgentTask(name="audio_task", agent="audio_agent", inputs=[], outputs=["instruction"], is_queue=False, func=lambda agent: agent.act()),
        AgentTask(name="language_task", agent="language_agent", inputs=["instruction"], outputs=["language_instruction"], is_queue=False, func=lambda agent, instr: agent.act(instr)),
        AgentTask(name="motor_task", agent="motor_agent", inputs=["language_instruction", "observation"], outputs=["motion"], is_queue=True, func=lambda agent, lang_instr, obs: agent.act(lang_instr, obs)),
        AgentTask(name="robot_task", agent="robot", inputs=["motion", "observation"], outputs=[], is_queue=False, func=lambda robot, motion: robot.do(motion)),
    ]

    hri = HRI(agents, tasks)
    hri.run()
