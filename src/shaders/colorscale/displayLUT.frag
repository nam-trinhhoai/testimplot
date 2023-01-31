#version 330 core

// vertices datas
in vec2 v_textureCoordinates;

// uniforms
uniform sampler1D f_colorMap; 
out vec4 f_fragColor; // shader output color

void main()
{ 
	f_fragColor = texture(f_colorMap, v_textureCoordinates.st.y);
	
    
}