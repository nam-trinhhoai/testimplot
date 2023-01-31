 #version 150 core

in vec3 vertexPosition;
in vec3 vertexNormal;
in vec2 vertexTexCoord;

uniform usampler2D surfaceMap;

out vec3 position;
out vec3 normal;
out vec2 texCoord;
out float potentialHole;
//out float height;

uniform mat4 modelMatrix;
//uniform mat4 modelView;
//uniform mat3 modelViewNormal;
uniform mat4 mvp;

uniform float heightThreshold;

void main()
{
	texCoord = vertexTexCoord;

	float origval = texture(surfaceMap, texCoord.st).r;
	if(origval < heightThreshold) {
		potentialHole = 0.0;
	} else {
		potentialHole = 1.0;
	}

	position = vec3(modelMatrix * vec4(vertexPosition, 1.0));//modelMatrix	
	normal = mat3(transpose(inverse(modelMatrix)))*vertexNormal;

    gl_Position = mvp * vec4( vertexPosition, 1.0 );
}
