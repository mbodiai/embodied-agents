import numpy as np
from mbodied.types.sense.world import WorldObject, World

def world_object_to_vector(world_object: WorldObject) -> np.ndarray:
    """
    Converts a WorldObject to a vector representation.
    
    Args:
        world_object (WorldObject): The object in the world to be converted.
        
    Returns:
        np.ndarray: A vector representing the object's state.
    """
    vector = []

    if world_object.bbox_2d:
        vector.extend([world_object.bbox_2d.x1, world_object.bbox_2d.y1, 
                       world_object.bbox_2d.x2, world_object.bbox_2d.y2])
    else:
        vector.extend([0, 0, 0, 0])

    if world_object.bbox_3d:
        vector.extend([world_object.bbox_3d.x1, world_object.bbox_3d.y1, world_object.bbox_3d.z1,
                       world_object.bbox_3d.x2, world_object.bbox_3d.y2, world_object.bbox_3d.z2])
    else:
        vector.extend([0, 0, 0, 0, 0, 0])

    if world_object.pose:
        vector.extend([
            world_object.pose.x, 
            world_object.pose.y,
            world_object.pose.z,
            world_object.pose.roll,
            world_object.pose.pitch,
            world_object.pose.yaw
        ])
    else:
        vector.extend([0, 0, 0, 0, 0, 0, 0])

    if world_object.pixel_coords:
        vector.extend([world_object.pixel_coords.u, world_object.pixel_coords.v])
    else:
        vector.extend([0, 0])

    return np.array(vector)


def world_to_vector(world: World, include_image: bool = False, include_depth: bool = False, include_annotated: bool = False) -> np.ndarray:
    """
    Converts the entire World object to a vector representation by 
    processing each WorldObject.
    
    Args:
        world (World): The world to be converted.
        include_image (bool): If True, include the image array in the vector. Defaults to False.
        include_depth (bool): If True, include the depth array in the vector. Defaults to False.
        include_annotated (bool): If True, include the annotated array in the vector. Defaults to False.
        
    Returns:
        np.ndarray: A vector representing the state of the world.
    """
    world_vector = []

    if include_image and world.image:
        world_vector.extend(world.image.array.flatten())
    if include_depth and world.depth:
        world_vector.extend(world.depth.array.flatten()) 
    if include_annotated and world.annotated:
        world_vector.extend(world.annotated.array.flatten())

    for obj in world.objects:
        obj_vector = world_object_to_vector(obj)
        world_vector.extend(obj_vector)

    return np.array(world_vector)
