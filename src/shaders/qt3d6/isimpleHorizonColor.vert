#version 150 core

in vec3 vertexPosition;
in vec3 vertexNormal;
in vec2 vertexTexCoord;

out vec3 worldPos;
//out vec3 viewPos;
//out vec3 position;
out vec3 normal;
//out vec3 worldNormal;
out vec2 texCoord;
//out vec3 lightA;
//out vec3 lightB;
out float potentialHole;
out float height;

uniform mat4 modelMatrix;
uniform mat4 modelView;
uniform mat3 modelViewNormal;
uniform mat3 modelNormal;
uniform mat4 mvp;
uniform mat4 viewMatrix;
uniform mat4 worldMatrix;

uniform isampler2D surfaceMap;
uniform float cubeOrigin;
uniform float cubeScale;
uniform float heightThreshold;

uniform float zScale;

float rand(float n){return fract(sin(n) * 43758.5453123);}



float getHeight(vec2 texC)
{
   return texture(surfaceMap, texC.st).r;
}


float texture_moyenne(in isampler2D t, in vec2 uv, in vec2 textureSize, in vec2 texelSize)
{
	//calcul de la moyenne en allant chercher les pixels autour

    float c = texture(t, uv).r;
    float c1 = texture(t, uv + vec2(texelSize.x, 0.0)).r ;
    float c2 = texture(t, uv + vec2(0.0, texelSize.y)).r ;
    float c3 = texture(t, uv + vec2(-texelSize.x, 0.0)).r ;
    float c4 = texture(t, uv + vec2(0.0, -texelSize.y)).r ;

    float c5 = texture(t, uv + vec2(-texelSize.x, -texelSize.y)).r ;
    float c6 = texture(t, uv + vec2(-texelSize.x, texelSize.y)).r ;
    float c7 = texture(t, uv + vec2(texelSize.x, -texelSize.y)).r ;
    float c8 = texture(t, uv + vec2(texelSize.x, texelSize.y)).r ;

    /*float c9 = texture(t, uv + vec2(2.0*texelSize.x, 0.0)).r ;
    float c10 = texture(t, uv + vec2(0.0, 2.0*texelSize.y)).r ;
    float c11 = texture(t, uv + vec2(-2.0*texelSize.x, 0.0)).r ;
    float c12 = texture(t, uv + vec2(0.0, -2.0* texelSize.y)).r ;*/
    return (c +c1+c2+c3+c4+ c5 +c6+c7+c8 )/9.0f;

}

void main()
{
	texCoord = vertexTexCoord;
	ivec2 texSize = textureSize(surfaceMap, 0);
	vec2 texelSize =vec2(1.0 / texSize);
	float origval = texture(surfaceMap, texCoord.st).r;
	//float origval = texture_moyenne(surfaceMap, texCoord.st,texSize,texelSize);

	//height = origval;
	if(origval < heightThreshold) {
		potentialHole = 0.0;
	} else {
		potentialHole = 1.0;
	}


	// compute normal from heightMap
	/*float x = getHeight(vec2(texCoord.s - texelSize.x, texCoord.t)) - getHeight(vec2(texCoord.s + texelSize.x, texCoord.t));
	float y = 2.0;
	float z = getHeight(vec2(texCoord.s, texCoord.t + texelSize.y)) - getHeight(vec2(texCoord.s, texCoord.t - texelSize.y));

	vec3 normalH = normalize(vec3(-x, -y, -z) / 2.0);		


	vec3 newPos = vertexPosition;
	newPos.y = cubeOrigin + cubeScale * origval;
*/
normal = mat3(transpose(inverse(modelMatrix)))*vertexNormal;
	worldPos = vec3(modelMatrix * vec4(vertexPosition, 1.0));//modelMatrix
	//worldNormal =mat3(transpose(inverse(modelMatrix)))*normalH;
	gl_Position = mvp * vec4(vertexPosition, 1.0);
}
