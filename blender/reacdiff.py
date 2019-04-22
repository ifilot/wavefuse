import bpy
import sys
import os

# read input and output file
argv = sys.argv
argv = argv[argv.index("--") + 1:]
inputfile = argv[0]
outputfile = argv[1]

imported_object = bpy.ops.import_mesh.ply(filepath=inputfile)
obj_object = bpy.context.selected_objects[0]
bpy.ops.object.shade_smooth()
mat = bpy.data.materials.get('shiny')
obj_object.data.materials.append(mat)
scale = 0.25
obj_object.scale.x = scale
obj_object.scale.y = scale
obj_object.scale.z = scale
obj_object.location.x = 0
obj_object.location.y = 0
obj_object.location.z = 0

for scene in bpy.data.scenes:
	scene.cycles.device = 'GPU'

bpy.data.scenes['Scene'].render.filepath = outputfile
bpy.ops.render.render(write_still=True)
