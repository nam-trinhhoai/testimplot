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

	//normal = vertexNormal;
	//position= vertexPosition;
	worldPosition = vec3(modelMatrix * vec4(vertexPosition, 1.0));
	
	worldNormal = normalize(modelNormalMatrix*vertexNormal);
    gl_Position = mvp*vec4(vertexPosition, 1.0);
}





/*
#version 430 core

in vec3 vertexPosition;  // Qt3D default name (see qattribute.cpp)
in vec3 vertexNormal;
in vec3 vertexColor;
//in vec3 vertexTexCoord;
//in vec3 vertexTangent;

out vec3 position;
out vec3 col;
out vec3 norm;

uniform mat4 modelView;
uniform mat4 mvp;

uniform mat4 normalMatrix;
uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;


void main()
{
 //   vec3 t = vec3(vertexTexCoord, 1.0);
 //   texCoord = (t / t.z).xy;
  //  position = vertexColor;//vec3(modelView * vec4(vertexPosition, 1.0));
    col = vec3(0,0,1);//vertexColor;
 //   norm = (normalMatrix*vec4(vertexNormal, 0.0)).xyz;
     norm = vertexNormal;


    // position = vertexPosition;
    //gl_Position = mvp * vec4(vertexPosition, 1.0);
    //mat4 all = projectionMatrix*modelMatrix;
    mat4 all = projectionMatrix*viewMatrix*modelMatrix;

    col = (modelMatrix*vec4(vertexPosition, 1.0)).zzz/5;
    gl_Position = all*vec4(vertexPosition, 1.0);

    position = gl_Position.xyz;
}
*/

/*
#version 150 core

in vec3 vertexPosition;  // Qt3D default name (see qattribute.cpp)
in vec3 vertexNormal;
in vec3 vertexColor;


out vec3 position;
//out vec3 col;
out vec3 normal;


uniform mat4 mvp;
//uniform mat4 normalMatrix;

//uniform mat4 modelMatrix;
//uniform mat4 viewMatrix;
//uniform mat4 projectionMatrix;

void main()
{
   // normal = (normalMatrix*vec4(vertexNormal, 0.0)).xyz;
//   mat4 all = projectionMatrix*viewMatrix*modelMatrix;
    gl_Position = mvp*vec4(vertexPosition, 1.0);
}

*/

/*
#version 430 core

in vec3 vertexPosition;  // Qt3D default name (see qattribute.cpp)
in vec3 vertexNormal;
in vec3 vertexColor;
//in vec3 vertexTexCoord;
//in vec3 vertexTangent;

//out vec3 position;
out vec3 col;
out vec3 norm;

uniform mat4 modelView;
uniform mat4 mvp;

uniform mat4 normalMatrix;
uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projectionMatrix;


void main()
{
 //   vec3 t = vec3(vertexTexCoord, 1.0);
 //   texCoord = (t / t.z).xy;
  //  position = vertexColor;//vec3(modelView * vec4(vertexPosition, 1.0));
  //  col = vec3(0,0,1);//vertexColor;
   // position = vertexPosition;
    //gl_Position = mvp * vec4(vertexPosition, 1.0);
    //mat4 all = projectionMatrix*modelMatrix;
    mat4 all = projectionMatrix*viewMatrix*modelMatrix;
     norm = vertexNormal;

  //  col = (modelMatrix*vec4(vertexPosition, 1.0)).zzz/5;
    gl_Position = all*vec4(vertexPosition, 1.0);

  //  position = gl_Position.xyz;
}
*/