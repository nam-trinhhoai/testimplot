#ifndef SURFACEMESH_HPP
#define SURFACEMESH_HPP

#include <QFileInfo>

#include <cmath>

template<typename IsoType>
void SurfaceMesh::createCache(SurfaceMeshCache& cache,const IsoType* isobuffer ,int width, int depth ,float steps, float cubeOrigin,float cubeScale,QString path, QMatrix4x4 transform, int compression)
{

	createObj<IsoType>(path,isobuffer,width,depth,steps, cubeOrigin,cubeScale,transform,compression);
	loadObj(path.toStdString().c_str(),cache);

}

template<typename IsoType>
void SurfaceMesh::createObj(QString path,const IsoType* isobuffer, int width, int height, float steps, float cubeOrigin,float cubeScale,QMatrix4x4 objectTransform, int compression)
{

	qDebug()<<"OBSOLETE WARNING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";

/*	QFileInfo fileInfo(path);
	//vertices.clear();
	//triangles.clear();
	 QVector<Triangle> triangles;
	 QVector<Vertex> vertices;

	if( !fileInfo.exists())
	{

		//SurfaceGenerator surf(m_dimensions.x()/s,m_dimensions.y()/s, transform);
		int numVertex=(width) * (height);//  surf.getVerticesCount();


	//	palette->lockPointer();
	//	short* tab = static_cast<short*>(palette->backingPointer());

		int textureWidth = width;
		int textureHeight = height;

		int meshWidth = (width-1)/steps+1;//getWidth();
		 int meshHeight =(height-1)/steps+1;//getHeight();

		// qDebug()<<" create obj : "<<meshWidth<<" , " <<meshHeight;

	for (int row = 0; row < meshWidth; row+=1) {
		for (int col = 0; col < meshHeight; col+=1) {

			int bufferIndexMesh = col * meshWidth + row;

			int textureIndexWidth = ((float) row / (float ) meshWidth) * (float) textureWidth;
			textureIndexWidth = std::min(textureIndexWidth, textureWidth - 1);
			int textureIndexHeight = ((float) col / (float ) meshHeight) * (float) textureHeight;
			textureIndexHeight = std::min(textureIndexHeight, textureHeight - 1);


			double heightValue = isobuffer[textureIndexWidth + textureIndexHeight * textureWidth];

			float newHeightValue =  cubeOrigin + cubeScale *heightValue;
			QVector3D inVect(row*steps, newHeightValue, col*steps);
			QVector3D outVect = objectTransform * inVect;
		//	long iv = (col + row * m_width) * 3;
		//	long it = (col + row * m_width) * 2;
		//	long in = (col + row * m_width) * 3;

			Vertex vert;
			vert.p = outVect;
			vert.n = QVector3D(0.0,-1.0,0.0);

			//QVector2D uv( col*1.0/(meshHeight-1),row*1.0/(meshWidth-1));
			QVector2D uv(row*1.0/(meshWidth-1), col*1.0/(meshHeight-1));
			vert.uvs = uv;
			//uvs.push_back(uv);
			vertices.push_back(vert);

		}
	}


	int sensIndex;
	const float*transfo = objectTransform.constData();
	float xscale = transfo[0];
	float zscale = transfo[10];

	if((xscale > 0 && zscale > 0) || (xscale < 0 && zscale < 0))
	{
		sensIndex = 1;
	}else{
		sensIndex = -1;
	}
	if (sensIndex>0) {
		for (int row = 0; row < meshWidth-1; row+=1) {
			for (int col = 0; col < meshHeight-1; col+=1) {
				//long i = (col + row * (meshHeight-1)) * 6;
				Triangle t1;

				t1.v[0] = col + row * meshHeight;
				t1.v[1] = col + 1 + row * meshHeight;
				t1.v[2] = col + (row + 1) * meshHeight;

				t1.uvs[0] =  vertices[t1.v[0]].uvs; //uvs[t1.v[0]];
				t1.uvs[1] =  vertices[t1.v[1]].uvs; //uvs[t1.v[1]];
				t1.uvs[2] =  vertices[t1.v[2]].uvs; // uvs[t1.v[2]];

				//QVector3D normal = QVector3D::crossProduct(p[1]-p[0],p[2]-p[0]); //n.cross(p[1]-p[0],p[2]-p[0]);
				QVector3D normal = QVector3D::crossProduct(vertices[t1.v[2]].p-vertices[t1.v[0]].p,vertices[t1.v[1]].p- vertices[t1.v[0]].p);
				normal.normalize();
				t1.n = normal;//QVector3D(0.0,1.0,0.0);
				triangles.push_back(t1);

				Triangle t2;
				t2.v[0] = col + (row + 1) * meshHeight;
				t2.v[1] = col + 1 + row * meshHeight;
				t2.v[2] = col + 1 + (row + 1) * meshHeight;

				t2.uvs[0] =  vertices[t2.v[0]].uvs;
				t2.uvs[1] =  vertices[t2.v[1]].uvs;
				t2.uvs[2] =  vertices[t2.v[2]].uvs;

				QVector3D normal2 = QVector3D::crossProduct(vertices[t2.v[2]].p-vertices[t2.v[0]].p,vertices[t2.v[1]].p- vertices[t2.v[0]].p);
				normal2.normalize();
				t2.n = normal2;



				//t2.n = QVector3D(0.0,1.0,0.0);

				triangles.push_back(t2);

			}
		}
	} else {
		for (int row = 0; row < meshWidth-1; row+=1) {
			for (int col = 0; col < meshHeight-1; col+=1) {
				//long i = (col + row * (meshHeight-1)) * 6;
				Triangle t1;

				t1.v[0] = col + row * meshHeight;
				t1.v[1] = col + (row + 1) * meshHeight;
				t1.v[2] = col + 1 + row * meshHeight;

				t1.uvs[0] =  vertices[t1.v[0]].uvs; //uvs[t1.v[0]];
				t1.uvs[1] =  vertices[t1.v[1]].uvs; //uvs[t1.v[1]];
				t1.uvs[2] =  vertices[t1.v[2]].uvs; // uvs[t1.v[2]];

				//QVector3D normal = QVector3D::crossProduct(p[1]-p[0],p[2]-p[0]); //n.cross(p[1]-p[0],p[2]-p[0]);
				QVector3D normal = QVector3D::crossProduct(vertices[t1.v[2]].p-vertices[t1.v[0]].p,vertices[t1.v[1]].p- vertices[t1.v[0]].p);
				normal.normalize();
				t1.n = normal;//QVector3D(0.0,1.0,0.0);
				triangles.push_back(t1);

				Triangle t2;
				t2.v[0] = col + (row + 1) * meshHeight;
				t2.v[1] = col + 1 + (row + 1) * meshHeight;
				t2.v[2] = col + 1 + row * meshHeight;

				t2.uvs[0] =  vertices[t2.v[0]].uvs;
				t2.uvs[1] =  vertices[t2.v[1]].uvs;
				t2.uvs[2] =  vertices[t2.v[2]].uvs;

				QVector3D normal2 = QVector3D::crossProduct(vertices[t2.v[2]].p-vertices[t2.v[0]].p,vertices[t2.v[1]].p- vertices[t2.v[0]].p);
				normal2.normalize();
				t2.n = normal2;



				//t2.n = QVector3D(0.0,1.0,0.0);

				triangles.push_back(t2);

			}
		}
	}

	if(compression > 0)
	{

		int nbsommet = vertices.size();
		int newNbSommet = (100- compression)*0.01f *nbsommet;
		//simplifyMesh( newNbSommet,true);
	}
	for(int i=0;i<triangles.size();i++)
		{

			int indexVertex1 = triangles[i].v[0];
			vertices[indexVertex1].n += triangles[i].n;
			vertices[indexVertex1].uvs.setX(triangles[i].uvs[0].x());
			vertices[indexVertex1].uvs.setY(triangles[i].uvs[0].y());
			int indexVertex2 = triangles[i].v[1];
			vertices[indexVertex2].n += triangles[i].n;
			vertices[indexVertex2].uvs.setX(triangles[i].uvs[1].x());
			vertices[indexVertex2].uvs.setY(triangles[i].uvs[1].y());
			int indexVertex3 = triangles[i].v[2];
			vertices[indexVertex3].n += triangles[i].n;
			vertices[indexVertex3].uvs.setX(triangles[i].uvs[2].x());
			vertices[indexVertex3].uvs.setY(triangles[i].uvs[2].y());
		}

		for(int i=0;i<vertices.size();i++)
		{
			vertices[i].n =vertices[i].n.normalized();
		}


	writeObj(path.toStdString().c_str(),triangles,vertices);

	}*/
}

#endif // SURFACEMESH_HPP
