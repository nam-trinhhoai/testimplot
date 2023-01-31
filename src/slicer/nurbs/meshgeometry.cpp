#include "meshgeometry.h"
#include <Qt3DCore/QBuffer>
#include <Qt3DCore/QAttribute>
#include <Qt3DCore/QGeometry>
//#include <QDebug>
#include <QVector3D>
#include <QDateTime>

MeshGeometry::MeshGeometry(Qt3DCore::QNode *parent)
    : Qt3DRender::QGeometryRenderer(parent)
{
    // todo: what must be deleted and what deletes itself?

    m_geometry     = new Qt3DCore::QGeometry(this);
    m_vertexBuffer = new Qt3DCore::QBuffer(m_geometry);
    m_indexBuffer  = new Qt3DCore::QBuffer(m_geometry);

    m_positionAttribute = new Qt3DCore::QAttribute(parent);
    m_normalAttribute   = new Qt3DCore::QAttribute(parent);
   // m_colorAttribute    = new Qt3DRender::QAttribute(parent);
    m_indexAttribute    = new Qt3DCore::QAttribute(parent);

    int valuesPerPosition = 3, valuesPerNormal = 3, valuesPerColor = 3;
    m_valuesPerVertex = valuesPerPosition + valuesPerNormal;// + valuesPerColor; // vertices(3) + normals(3)  + colors(3)
    const quint32 stride = m_valuesPerVertex* sizeof(float);

    uint byteoffset = 0, vertexsize = 0;
    // Attribute setup

    vertexsize = valuesPerPosition;
    m_positionAttribute->setName(Qt3DCore::QAttribute::defaultPositionAttributeName());
    m_positionAttribute->setVertexBaseType(Qt3DCore::QAttribute::Float);
    m_positionAttribute->setVertexSize(vertexsize);
    m_positionAttribute->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
    m_positionAttribute->setBuffer(m_vertexBuffer);
    m_positionAttribute->setByteStride(stride);
    m_positionAttribute->setByteOffset(byteoffset);
    m_positionAttribute->setCount(0);
    byteoffset += vertexsize;

    vertexsize = valuesPerNormal;
    m_normalAttribute->setName(Qt3DCore::QAttribute::defaultNormalAttributeName());
    m_normalAttribute->setVertexBaseType(Qt3DCore::QAttribute::Float);
    m_normalAttribute->setVertexSize(vertexsize);
    m_normalAttribute->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
    m_normalAttribute->setBuffer(m_vertexBuffer);
    m_normalAttribute->setByteStride(stride);
    m_normalAttribute->setByteOffset(byteoffset * sizeof(float));
    m_normalAttribute->setCount(0);
    byteoffset += vertexsize;

  /*  vertexsize = valuesPerColor;
    m_colorAttribute->setName(Qt3DRender::QAttribute::defaultColorAttributeName());
    m_colorAttribute->setVertexBaseType(Qt3DRender::QAttribute::Float);
    m_colorAttribute->setVertexSize(vertexsize);
    m_colorAttribute->setAttributeType(Qt3DRender::QAttribute::VertexAttribute);
    m_colorAttribute->setBuffer(m_vertexBuffer);
    m_colorAttribute->setByteStride(stride);
    m_colorAttribute->setByteOffset(byteoffset * sizeof(float));
    m_colorAttribute->setCount(0);
    byteoffset += vertexsize;*/

    m_indexAttribute->setAttributeType(Qt3DCore::QAttribute::IndexAttribute);
    m_indexAttribute->setVertexBaseType(Qt3DCore::QAttribute::UnsignedInt);
    m_indexAttribute->setBuffer(m_indexBuffer);
    m_indexAttribute->setCount(0);
  //  m_indexAttribute->setVertexSize(1);
  //  m_indexAttribute->setByteOffset(0);
  //  m_indexAttribute->setByteStride(0);

    m_geometry->addAttribute(m_positionAttribute);
    m_geometry->addAttribute(m_normalAttribute);
 //   m_geometry->addAttribute(m_colorAttribute);
    m_geometry->addAttribute(m_indexAttribute);

   // setInstanceCount(1);
   // setIndexOffset(0);
   // setFirstInstance(0);
    setPrimitiveType(Qt3DRender::QGeometryRenderer::Triangles);

   // setPrimitiveType(Qt3DRender::QGeometryRenderer::Lines);

    setGeometry(m_geometry);

/*
    Vec3 startTop(0,0,1), startBottom(0,0,0), endTop(0,1,1), endBottom(0,1,0);
    std::vector<Vec3> vertices={startBottom, startTop, endTop, endBottom};
    std::vector<int> indices={0,1,2, 2,3,0};
    std::vector<Vec3> normals ={Vec3(1,0,0), Vec3(1,0,0), Vec3(1,0,0), Vec3(1,0,0)};
    std::vector<Vec3> colors  ={Vec3(0,0,0), Vec3(0,1,0), Vec3(1,1,0), Vec3(1,0,0)};
    uploadMeshData(vertices, indices, normals, colors);*/

}

MeshGeometry::~MeshGeometry()
{
   //  std::cout << "~CRendererMesh()" << std::endl;
}

void MeshGeometry::setRenderPoints(bool pointsNotTriangles)
{
    if (pointsNotTriangles)  setPrimitiveType(Qt3DRender::QGeometryRenderer::Points);
    else                     setPrimitiveType(Qt3DRender::QGeometryRenderer::Triangles);
}

void MeshGeometry::setRenderLines(bool linesNotTriangles)
{
    if (linesNotTriangles)  setPrimitiveType(Qt3DRender::QGeometryRenderer::Lines);
    else                    setPrimitiveType(Qt3DRender::QGeometryRenderer::Triangles);
}

void MeshGeometry::setEmptyData()
{
    std::vector<QVector3D> vertices;
    std::vector<int> indices;
    std::vector<QVector3D> normals;
 //   std::vector<QVector3D> colors;
    uploadMeshData(vertices, indices, normals);//, colors);
}

void MeshGeometry::uploadMeshData(std::vector<QVector3D>& vertices, std::vector<int>& indices, std::vector<QVector3D>& normals)//, std::vector<QVector3D>& colors)
{
    // vec3 for position
    // vec3 for colors
    // vec3 for normals

    size_t numVertices = vertices.size();
    size_t numIndices  = indices.size();

    QVector<QVector3D> verts;
    for (size_t i = 0; i <numVertices; i++)   
        verts << vertices[i] << normals[i];// << colors[i];


    QByteArray vertexBufferData;
    vertexBufferData.resize(uint(numVertices) * m_valuesPerVertex * sizeof(float));
    float *rawVertexArray = reinterpret_cast<float *>(vertexBufferData.data());

    uint ct = 0;
    for (const QVector3D &v : verts)
    {
        rawVertexArray[ct++] = v.x();
        rawVertexArray[ct++] = v.y();
        rawVertexArray[ct++] = v.z();
    }


    QByteArray indexBufferData;
    indexBufferData.resize(uint(numIndices) * sizeof(unsigned int));
    unsigned int *rawIndexArray = reinterpret_cast<unsigned int *>(indexBufferData.data());


    ct = 0;
    for (uint i = 0; i <numIndices; i++)
    {
        rawIndexArray[ct++] = indices[i];
    }

    m_indexAttribute->setCount(uint(numIndices));
    m_indexBuffer->setData(indexBufferData);

    m_vertexBuffer->setData(vertexBufferData);
    m_positionAttribute->setCount(uint(numVertices));
      m_normalAttribute->setCount(uint(numVertices));
     //  m_colorAttribute->setCount(uint(numVertices));

 //   setVertexCount(numIndices);
}


//-------------------------------------------------------------------------------------------------------

MeshGeometry2::MeshGeometry2(Qt3DCore::QNode *parent)
    : Qt3DRender::QGeometryRenderer(parent)
{

    // todo: what must be deleted and what deletes itself?
    m_geometry     = new Qt3DCore::QGeometry(this);

    m_vertexBuffer = new Qt3DCore::QBuffer(m_geometry);
    m_indexBuffer  = new Qt3DCore::QBuffer(m_geometry);
    m_normalBuffer = new Qt3DCore::QBuffer(m_geometry);

    m_positionAttribute = new Qt3DCore::QAttribute(parent);
    m_normalAttribute   = new Qt3DCore::QAttribute(parent);
    m_indexAttribute    = new Qt3DCore::QAttribute(parent);

    int valuesPerPosition = 3, valuesPerNormal = 3;
   // m_valuesPerVertex = valuesPerPosition + valuesPerNormal;// + valuesPerColor; // vertices(3) + normals(3)  + colors(3)
   // const quint32 stride = m_valuesPerVertex* sizeof(float);

    uint byteoffset = 0, vertexsize = 0;
    // Attribute setup



   // m_vertexBuffer->setType(Qt3DCore::QBuffer::VertexBuffer);
	m_positionAttribute->setName(Qt3DCore::QAttribute::defaultPositionAttributeName());
	m_positionAttribute->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
//	m_positionAttribute->setDataType(Qt3DCore::QAttribute::Float);
	m_positionAttribute->setBuffer(m_vertexBuffer);
	m_positionAttribute->setVertexSize(3);
//	m_positionAttribute->setDataSize(3);
	m_positionAttribute->setByteOffset(0);
	m_positionAttribute->setByteStride(3 * sizeof(float));
	m_positionAttribute->setCount(0);


	//m_normalBuffer->setType(Qt3DCore::QBuffer::VertexBuffer);
	m_normalAttribute->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
	m_normalAttribute->setBuffer(m_normalBuffer);
	//m_normalAttribute->setDataType(Qt3DCore::QAttribute::Float);
	//m_normalAttribute->setDataSize(3);
	m_normalAttribute->setVertexSize(3);
	m_normalAttribute->setByteOffset(0);
	m_normalAttribute->setByteStride(0);
	m_normalAttribute->setCount(0);
	m_normalAttribute->setName(Qt3DCore::QAttribute::defaultNormalAttributeName());

//	m_indexBuffer->setType(Qt3DCore::QBuffer::UniformBuffer);
	m_indexAttribute->setAttributeType(Qt3DCore::QAttribute::IndexAttribute);
	m_indexAttribute->setBuffer(m_indexBuffer);
	  m_indexAttribute->setVertexBaseType(Qt3DCore::QAttribute::UnsignedInt);
	//m_indexAttribute->setDataType(Qt3DCore::QAttribute::UnsignedInt);
	//m_indexAttribute->setDataSize(1);
	m_indexAttribute->setByteOffset(0);
	m_indexAttribute->setByteStride(3 * sizeof(unsigned int));
	m_indexAttribute->setCount(0);

	m_geometry->addAttribute(m_positionAttribute);
	m_geometry->addAttribute(m_normalAttribute);

	m_geometry->addAttribute(m_indexAttribute);

	setPrimitiveType(Qt3DRender::QGeometryRenderer::Triangles);//Triangles);//Lines
	setGeometry(m_geometry);

   /* vertexsize = valuesPerPosition;
    m_positionAttribute->setName(Qt3DRender::QAttribute::defaultPositionAttributeName());
    m_positionAttribute->setVertexBaseType(Qt3DRender::QAttribute::Float);
    m_positionAttribute->setVertexSize(vertexsize);
    m_positionAttribute->setAttributeType(Qt3DRender::QAttribute::VertexAttribute);
    m_positionAttribute->setBuffer(m_vertexBuffer);
    m_positionAttribute->setByteStride(stride);
    m_positionAttribute->setByteOffset(byteoffset);
    m_positionAttribute->setCount(0);
    byteoffset += vertexsize;

    vertexsize = valuesPerNormal;
    m_normalAttribute->setName(Qt3DRender::QAttribute::defaultNormalAttributeName());
    m_normalAttribute->setVertexBaseType(Qt3DRender::QAttribute::Float);
    m_normalAttribute->setVertexSize(vertexsize);
    m_normalAttribute->setAttributeType(Qt3DRender::QAttribute::VertexAttribute);
    m_normalAttribute->setBuffer(m_vertexBuffer);
    m_normalAttribute->setByteStride(stride);
    m_normalAttribute->setByteOffset(byteoffset * sizeof(float));
    m_normalAttribute->setCount(0);
    byteoffset += vertexsize;



    m_indexAttribute->setAttributeType(Qt3DRender::QAttribute::IndexAttribute);
    m_indexAttribute->setVertexBaseType(Qt3DRender::QAttribute::UnsignedInt);
    m_indexAttribute->setBuffer(m_indexBuffer);
    m_indexAttribute->setCount(0);

    m_geometry->addAttribute(m_positionAttribute);
    m_geometry->addAttribute(m_normalAttribute);
 //   m_geometry->addAttribute(m_colorAttribute);
    m_geometry->addAttribute(m_indexAttribute);

   // setInstanceCount(1);
   // setIndexOffset(0);
   // setFirstInstance(0);
    setPrimitiveType(Qt3DRender::QGeometryRenderer::Triangles);



    setGeometry(m_geometry);*/



}

MeshGeometry2::~MeshGeometry2()
{

}


void MeshGeometry2::setEmptyData()
{
    std::vector<QVector3D> vertices;
    std::vector<int> indices;
    std::vector<QVector3D> normals;

    uploadMeshData(vertices, indices, normals);
}

void MeshGeometry2::setRenderLines(bool linesNotTriangles)
{
    if (linesNotTriangles)  setPrimitiveType(Qt3DRender::QGeometryRenderer::Lines);
    else                    setPrimitiveType(Qt3DRender::QGeometryRenderer::Triangles);
}

void MeshGeometry2::uploadMeshData(std::vector<QVector3D>& vertices, std::vector<int>& indices, std::vector<QVector3D>& normals)//, std::vector<QVector3D>& colors)
{
    // vec3 for position
    // vec3 for normals

    size_t numVertices = vertices.size();
    size_t numIndices  = indices.size();

  /*  QVector<QVector3D> verts;
    for (size_t i = 0; i <numVertices; i++)
        verts << vertices[i] << normals[i];// << colors[i];

*/
    QByteArray vertexBufferData;
    vertexBufferData.resize(uint(numVertices) * 3 * sizeof(float));
    float *rawVertexArray = reinterpret_cast<float *>(vertexBufferData.data());

    QByteArray normalBufferData;
    normalBufferData.resize(uint(numVertices) * 3 * sizeof(float));
    float *rawNormalArray = reinterpret_cast<float *>(normalBufferData.data());

    uint ct = 0;
    for (const QVector3D &v : vertices)
    {
        rawVertexArray[ct++] = v.x();
        rawVertexArray[ct++] = v.y();
        rawVertexArray[ct++] = v.z();
    }

    uint ctn = 0;
	for (const QVector3D &n : normals)
	{

		rawNormalArray[ctn++] = n.x();
		rawNormalArray[ctn++] = n.y();
		rawNormalArray[ctn++] = n.z();

	}

    QByteArray indexBufferData;
    indexBufferData.resize(uint(numIndices) * sizeof(unsigned int));
    unsigned int *rawIndexArray = reinterpret_cast<unsigned int *>(indexBufferData.data());


    ct = 0;
    for (uint i = 0; i <numIndices; i++)
    {
        rawIndexArray[ct++] = indices[i];
    }







    m_indexAttribute->setCount(uint(numIndices));
    m_indexBuffer->setData(indexBufferData);

    m_vertexBuffer->setData(vertexBufferData);
    m_positionAttribute->setCount(uint(numVertices));


   // computeNormals();

    m_normalBuffer->setData(normalBufferData);
    m_normalAttribute->setCount(uint(numVertices));

    //setVertexCount(numIndices);

}
/*
void MeshGeometry2::computeNotmals()
{
	for(int i=0;i<3*nbTri;i+=3)
	{
			QVector3D p1(rawVertexArray[rawIndexArray[i]*3],rawVertexArray[rawIndexArray[i]*3  +1],rawVertexArray[rawIndexArray[i]*3+2]);
			 QVector3D p2(rawVertexArray[rawIndexArray[i+1]*3],rawVertexArray[rawIndexArray[i+1]*3+1],rawVertexArray[rawIndexArray[i+1]*3+2]);
			 QVector3D p3(rawVertexArray[rawIndexArray[i+2]*3],rawVertexArray[rawIndexArray[i+2]*3+1],rawVertexArray[rawIndexArray[i+2]*3+2]);

			 QVector3D v1 = p3 - p1;  //3-2
			 QVector3D v2 = p2 - p1;  //1-2

			 QVector3D n1 = QVector3D::crossProduct(v1,v2);
			 n1 = n1.normalized();
	}

	for (int k=0; k<3; k++) {
				 rawNormalArray[rawIndexArray[i+k]*3] += n1.x();
				 rawNormalArray[rawIndexArray[i+k]*3+1] += n1.y();
				 rawNormalArray[rawIndexArray[i+k]*3+2] += n1.z();
			 }

}*/

void MeshGeometry2::computeNormals( )//float* rawVertexArray)//float *vertices, float * normals,unsigned int *indices)
{
	const auto *indices = reinterpret_cast< const uint*>(m_indexBuffer->data().constData());
//	auto *normals = reinterpret_cast<float*>(m_normalBuffer->data().data());
	const auto *rawVertexArray = reinterpret_cast<const float*>(m_vertexBuffer->data().constData());

	 int nbindice = m_indexAttribute->count();
	 int nbpoint = m_positionAttribute->count();


	 QByteArray normalBufferData;
	 normalBufferData.resize(3*nbpoint * sizeof(float));
	 normalBufferData.fill('\0');
	 auto *rawNormalArray = reinterpret_cast<float*>(normalBufferData.data());

	 for(int i=0;i< nbindice;i+=3)
	 {
		QVector3D p1(rawVertexArray[indices[i]*3],rawVertexArray[indices[i]*3  +1],rawVertexArray[indices[i]*3+2]);
		 QVector3D p2(rawVertexArray[indices[i+1]*3],rawVertexArray[indices[i+1]*3+1],rawVertexArray[indices[i+1]*3+2]);
		 QVector3D p3(rawVertexArray[indices[i+2]*3],rawVertexArray[indices[i+2]*3+1],rawVertexArray[indices[i+2]*3+2]);

		 QVector3D v1 = p3 - p2;
		 QVector3D v2 = p1 - p2;

		 QVector3D n1 = QVector3D::crossProduct(v1,v2);
		n1 = n1.normalized();

		rawNormalArray[indices[i]*3]+= n1.x();
		rawNormalArray[indices[i]*3+1]+= n1.y();
		rawNormalArray[indices[i]*3+2]+= n1.z();
		rawNormalArray[indices[i+1]*3]+= n1.x();
		rawNormalArray[indices[i+1]*3+1]+= n1.y();
		rawNormalArray[indices[i+1]*3+2]+= n1.z();
		rawNormalArray[indices[i+2]*3]+= n1.x();
		rawNormalArray[indices[i+2]*3+1]+= n1.y();
		rawNormalArray[indices[i+2]*3+2]+= n1.z();

	 }
	 m_normalBuffer->setData(normalBufferData);


/*	  QVector<QVector3D> listeNormals;
	    QVector<QVector3D> listeNormals2;
	    for(int i=0;i<normals.size()/2;i++)
	    {
	    	QVector3D N1 = vertices[i];
	    	QVector3D N2 = vertices[i]+normals[i]*250.0f;
	    	listeNormals.push_back(N1);
	    	listeNormals.push_back(N2);
	    }
	    for(int i=normals.size()/2;i<normals.size();i++)
	       {
	       	QVector3D N1 = vertices[i];
	       	QVector3D N2 = vertices[i]+normals[i]*200.0f;
	       	listeNormals2.push_back(N1);
	       	listeNormals2.push_back(N2);
	       }
	    Qt3DHelpers::drawNormals(listeNormals,Qt::green,this,1);
	    Qt3DHelpers::drawNormals(listeNormals2,Qt::red,this,1);

	    qDebug()<<"recompute normals";*/
}

