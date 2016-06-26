PROJECT(meshviewer)

# Status message
MESSAGE(STATUS "Configuring 'meshviewer'")

# Find includes in corresponding build directories
SET(CMAKE_INCLUDE_CURRENT_DIR ON)

# Instruct CMake to run moc automatically when needed.
SET(CMAKE_AUTOMOC ON)

# Find the Qt libraries
FIND_PACKAGE(Qt5 COMPONENTS Core Gui OpenGL Qml Quick)

# Do not automatically link Qt executables to qtmain target on Windows.
# Uses console mode only. Not need to link against qtmain.lib.
cmake_policy(SET CMP0020 OLD)

SET(INC
	gpusurfacemesh.h
	gpuvolumemesh.h
	meshview.h
	scene.h
)

SET(SRC
	gpusurfacemesh.cpp
	gpuvolumemesh.cpp
	main.cpp
	meshview.cpp
	scene.cpp
)

SET(QML
	main.qml

	CreateBarDialog.qml
	OpenMeshDialog.qml
)

SET(SHADERS
	shaders/3DSceneBindings.h

	shaders/plane.vert
	shaders/plane.geom
	shaders/trimesh.vert
	shaders/trimesh.geom
	shaders/tetramesh.vert
	shaders/tetramesh.geom
	shaders/mesh.frag
	
	shaders/objectid_tetramesh.vert
	shaders/objectid_tetramesh.geom
	shaders/objectid.frag
)

QT5_ADD_RESOURCES(RESOURCES meshviewer.qrc)

SOURCE_GROUP("qml" FILES ${QML})
SOURCE_GROUP("res" FILES ${RESOURCES})
SOURCE_GROUP("shaders" FILES ${SHADERS})
SOURCE_GROUP("" FILES ${SRC} ${INC})

ADD_EXECUTABLE(meshviewer ${SRC} ${INC} ${QML} ${RESOURCES} ${SHADERS})
SET_TARGET_PROPERTIES(meshviewer PROPERTIES FOLDER examples)

TARGET_LINK_LIBRARIES(meshviewer
	vcl_geometry
	vcl_graphics
	Qt5::Core
	Qt5::Gui
	Qt5::OpenGL
	Qt5::Qml
	Qt5::Quick
)
