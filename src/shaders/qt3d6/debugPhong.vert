VARYING vec3 vertexPosition;
VARYING vec3 vertexNormal;
VARYING vec2 vertexTexCoord;

uniform sampler2D surfaceMap;

VARYING vec3 position;
VARYING vec3 normal;
VARYING vec2 texCoord;
VARYING float potentialHole;

uniform mat4 modelMatrix;

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

    POSITION = mvp * vec4( vertexPosition, 1.0 );
}
