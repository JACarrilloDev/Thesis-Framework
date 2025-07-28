import bpy
import os

def export_scene_to_coppeliasim(filepath):
    """
    Exports the current Blender scene to a format compatible with CoppeliaSim.
    
    Args:
        filepath (str): The path where the exported file will be saved.
    """
    # Ensure the filepath has the correct extension
    if not filepath.endswith('.fbx'):
        filepath += '.fbx'
    
    # Set the export settings
    bpy.ops.export_scene.fbx(
        filepath=filepath,
        use_selection=False,
        apply_scale_options='FBX_SCALE_ALL',
        bake_space_transform=True,
        mesh_smooth_type='FACE',
        use_mesh_modifiers=True,
        use_custom_props=False,
        add_leaf_bones=False,
        use_armature_deform_only=True,
        use_tspace=True,
        use_triangles=True,
        use_vertex_groups=True,
        use_object_hierarchy=True,
        use_animation=False,
        use_subsurf=False,
        use_mesh_modifiers=True,
        use_normals=True,
        use_uvs=True,
        use_materials=True,
        use_custom_props=False
    )

def handle_collision_tags(objects):
    """
    Assigns collision tags to objects in the Blender scene for CoppeliaSim compatibility.
    
    Args:
        objects (list): List of Blender objects to process.
    """
    for obj in objects:
        if obj.type == 'MESH':
            # Example: Assign a collision tag based on the object's name
            if "collision" in obj.name.lower():
                obj["collision_tag"] = True
            else:
                obj["collision_tag"] = False

def main():
    # Example usage
    scene_filepath = os.path.join(os.getcwd(), "exported_scene.fbx")
    export_scene_to_coppeliasim(scene_filepath)
    handle_collision_tags(bpy.context.scene.objects)

if __name__ == "__main__":
    main()