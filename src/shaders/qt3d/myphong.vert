#version 150 core

in vec3 vertexPosition;  // Qt3D default name (see qattribute.cpp)
in vec3 vertexNormal;


out vec3 worldPosition;
out vec3 worldNormal;

uniform mat4 modelMatrix;
uniform mat3 modelNormalMatrix;

uniform mat4 mvp;

void main()
{

	worldPosition = vec3(modelMatrix * vec4(vertexPosition, 1.0));
	
	worldNormal = normalize(modelNormalMatrix*vertexNormal);
    gl_Position = mvp*vec4(vertexPosition, 1.0);
}