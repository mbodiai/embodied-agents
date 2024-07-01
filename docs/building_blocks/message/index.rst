Message
==========

Overview
^^^^^^^^^

The `Message <https://github.com/mbodiai/embodied-agents/blob/main/mbodied/types/message.py>`_ class represents a single completion sample space. It can be text, image, a list of text/images, Sample, or other modality. The Message class is designed to handle various types of content and supports different roles such as user, assistant, or system.

You can create a ``Message`` in versatile ways. They can all be understood by mbodi's backend.

.. code-block:: python
    
    Message(role="user", content="example text")
    Message(role="user", content=["example text", Image("example.jpg"), Image("example2.jpg")])
    Message(role="user", content=[Sample("Hello")])

