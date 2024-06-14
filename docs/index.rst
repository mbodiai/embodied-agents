.. Mbodied Agents documentation master file, created by
   sphinx-quickstart on Fri Jun  7 12:09:33 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

mbodied agents
=============================================

**mbodied agents** simplifies the integration of advanced AI models in robotics. It offers a unified platform for controlling various robots using state-of-the-art transformers and multimodal data processing. This toolkit enables experimentation with AI models, dataset collection and augmentation, and model training or finetuning for specific tasks. The goal is to develop intelligent, adaptable robots that learn from interactions and perform complex tasks in dynamic environments.

.. toctree::
   :maxdepth: 2

   overview/index
   installation/index
   getting_started/index
   glossary/index
   concepts/index
   building_blocks/index
   contributing/index
   .. system_components/index

.. raw:: html

   <div class="quickstart-guide">
     <div class="quickstart-text">
       <h2>Quickstart Guide</h2>
       <p>Start your path to understanding the Embodied Agents package with our quickstart guide.</p>
       <a class="btn" href="getting_started/index.html">Get started</a>
     </div>
     <div class="quickstart-code">
       <pre>
   <code class="python">
   
   SYSTEM_PROMPT = f"""
    You are robot with vision capabilities.
    For each task given, you respond in JSON format. Here's the JSON schema:
    {AnswerAndActionsList.model_json_schema()}
    """

   robot_agent = CognitiveAgent(context=SYSTEM_PROMPT, api_service="openai", api_key=openai_api_key)
   </code>
       </pre>
     </div>
   </div>




.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
