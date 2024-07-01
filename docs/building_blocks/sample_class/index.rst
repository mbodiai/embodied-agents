The Sample Class
=================

Overview
^^^^^^^^^^^^

The Sample class is a base model for serializing, recording, and manipulating arbitrary data. It is designed to be extensible, flexible, and strongly typed. By wrapping your observation or action objects in the `Sample <https://github.com/mbodiai/embodied-agents/blob/main/mbodied/base/sample.py>`_ class, you'll be able to convert to and from the following with ease:

- a Gym space for creating a new Gym environment
- a flattened list, array, or tensor for plugging into an ML model
- a HuggingFace dataset with semantic search capabilities
- a Pydantic BaseModel for reliable and quick json serialization/deserialization.

Creating a Sample
^^^^^^^^^^^^^^^^^^

Here is an example of creating a Sample and using its methods:

.. code-block:: python

   # Creating a Sample instance
   sample = Sample(observation=[1,2,3], action=[4,5,6])

   # Flattening the Sample instance
   flat_list = sample.flatten()
   print(flat_list) # Output: [1, 2, 3, 4, 5, 6]

   # Generating a simplified JSON schema
   schema = sample.schema()
   print(schema)
   # Output: {'type': 'object', 'properties': {'observation': {'type': 'array', 'items': {'type': 'integer'}}, 'action': {'type': 'array',   'items': {'type': 'integer'}}}}

   # Unflattening a list into a Sample instance
   unflattened_sample = Sample.unflatten(flat_list, schema)
   print(unflattened_sample) # Output: Sample(observation=[1, 2, 3], action=[4, 5, 6])
   
Serialization and Deserialization with Pydantic
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Sample class leverages Pydantic's powerful features for serialization and deserialization, allowing you to easily convert between Sample instances and JSON.

To serialize or deserialize a Sample instance with JSON:

.. code-block:: python

   # Serialize the Sample instance to JSON
   sample = Sample(observation=[1,2,3], action=[4,5,6])
   json_data = sample.model_dump_json()
   print(json_data) # Output: '{"observation": [1, 2, 3], "action": [4, 5, 6]}'

   # Deserialize the JSON data back into a Sample instance
   json_data = '{"observation": [1, 2, 3], "action": [4, 5, 6]}'
   sample = Sample.model_validate(from_json(json_data))
   print(sample) # Output: Sample(observation=[1, 2, 3], action=[4, 5, 6])

Converting to Different Containers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: python

   # Converting to a dictionary
   sample_dict = sample.to("dict")
   print(sample_dict) # Output: {'observation': [1, 2, 3], 'action': [4, 5, 6]}

   # Converting to a NumPy array
   sample_np = sample.to("np")
   print(sample_np) # Output: array([1, 2, 3, 4, 5, 6])

   # Converting to a PyTorch tensor
   sample_pt = sample.to("pt")
   print(sample_pt) # Output: tensor([1, 2, 3, 4, 5, 6])

   # Converting to a HuggingFace Dataset
   sample_hf = sample.to("hf")
   print(sample_hf)
   # Output: Dataset({
   #     features: ['observation', 'action'],
   #     num_rows: 3
   # })

Gym Space Integration
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python
   
   # Creating a Gym space from the Sample instance
   gym_space = sample.space()
   print(gym_space)
   # Output: Dict('action': Box(-inf, inf, (3,), float64), 'observation': Box(-inf, inf, (3,), float64))

See `sample.py <https://github.com/mbodiai/embodied-agents/blob/main/mbodied/base/sample.py>`_ for more details.