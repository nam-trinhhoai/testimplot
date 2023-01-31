
#include "randomview3d.h"
#include <Qt3DExtras/QPlaneMesh>

#include "randomlineview.h"

#include "randomtexdataset.h"
//#include "helperqt3d.h"
#include "qt3dhelpers.h"
//#include "meshgeometry.h"
#include <Qt3DCore/QBuffer>
#include <Qt3DCore/QAttribute>
#include <Qt3DCore/QGeometry>
#include <Qt3DRender/QParameter>
//#include <QDebug>
#include <QVector3D>
#include <QDateTime>

CustomGeometry::CustomGeometry(Qt3DCore::QNode *parent)
    : Qt3DRender::QGeometryRenderer(parent)
{
    // todo: what must be deleted and what deletes itself?
    m_geometry     = new Qt3DCore::QGeometry(this);
    m_vertexBuffer = new Qt3DCore::QBuffer(m_geometry);
    m_indexBuffer  = new Qt3DCore::QBuffer(m_geometry);
    m_texBuffer	= new Qt3DCore::QBuffer(m_geometry);

    m_positionAttribute = new Qt3DCore::QAttribute(parent);
  //  m_normalAttribute   = new Qt3DRender::QAttribute(parent);
  //  m_colorAttribute    = new Qt3DRender::QAttribute(parent);
    m_indexAttribute    = new Qt3DCore::QAttribute(parent);
    m_textureAttribute	= new Qt3DCore::QAttribute(parent);

    int valuesPerPosition = 3;//, valuesPerNormal = 3, valuesPerColor = 3;
    int valuesPerUV =2;

    m_valuesPerVertex = valuesPerPosition ;//+ valuesPerNormal + valuesPerColor; // vertices(3) + normals(3)  + colors(3)
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

   /* vertexsize = valuesPerNormal;
    m_normalAttribute->setName(Qt3DRender::QAttribute::defaultNormalAttributeName());
    m_normalAttribute->setVertexBaseType(Qt3DRender::QAttribute::Float);
    m_normalAttribute->setVertexSize(vertexsize);
    m_normalAttribute->setAttributeType(Qt3DRender::QAttribute::VertexAttribute);
    m_normalAttribute->setBuffer(m_vertexBuffer);
    m_normalAttribute->setByteStride(stride);
    m_normalAttribute->setByteOffset(byteoffset * sizeof(float));
    m_normalAttribute->setCount(0);
    byteoffset += vertexsize;

    vertexsize = valuesPerColor;
    m_colorAttribute->setName(Qt3DRender::QAttribute::defaultColorAttributeName());
    m_colorAttribute->setVertexBaseType(Qt3DRender::QAttribute::Float);
    m_colorAttribute->setVertexSize(vertexsize);
    m_colorAttribute->setAttributeType(Qt3DRender::QAttribute::VertexAttribute);
    m_colorAttribute->setBuffer(m_vertexBuffer);
    m_colorAttribute->setByteStride(stride);
    m_colorAttribute->setByteOffset(byteoffset * sizeof(float));
    m_colorAttribute->setCount(0);
    byteoffset += vertexsize;*/


 //  vertexsize = valuesPerUV;
  //  m_texBuffer->setType(Qt3DCore::QBuffer::VertexBuffer);
   	m_textureAttribute->setName(Qt3DCore::QAttribute::defaultTextureCoordinateAttributeName());
   	m_textureAttribute->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
   //	m_textureAttribute->setDataType(Qt3DCore::QAttribute::Float);
   	m_textureAttribute->setBuffer(m_texBuffer);
   	m_textureAttribute->setVertexSize(2);
  // 	m_textureAttribute->setDataSize(2);
   	m_textureAttribute->setByteOffset(0 * sizeof(float));
   	m_textureAttribute->setByteStride(0);
   	m_textureAttribute->setCount(0);
  // 	byteoffset += vertexsize;

   	m_indexAttribute->setVertexBaseType(Qt3DCore::QAttribute::UnsignedInt);
    m_indexAttribute->setAttributeType(Qt3DCore::QAttribute::IndexAttribute);
    m_indexAttribute->setBuffer(m_indexBuffer);
    m_indexAttribute->setCount(0);
  //  m_indexAttribute->setVertexSize(1);
  //  m_indexAttribute->setByteOffset(0);
  //  m_indexAttribute->setByteStride(0);




    m_geometry->addAttribute(m_positionAttribute);
   // m_geometry->addAttribute(m_normalAttribute);
  //  m_geometry->addAttribute(m_colorAttribute);
    m_geometry->addAttribute(m_textureAttribute);
    m_geometry->addAttribute(m_indexAttribute);

   // setInstanceCount(1);
   // setIndexOffset(0);
   // setFirstInstance(0);
    setPrimitiveType(Qt3DRender::QGeometryRenderer::Triangles);

 //   setPrimitiveType(Qt3DRender::QGeometryRenderer::Lines);

    setGeometry(m_geometry);




/*
    Vec3 startTop(0,0,1), startBottom(0,0,0), endTop(0,1,1), endBottom(0,1,0);
    std::vector<Vec3> vertices={startBottom, startTop, endTop, endBottom};
    std::vector<int> indices={0,1,2, 2,3,0};
    std::vector<Vec3> normals ={Vec3(1,0,0), Vec3(1,0,0), Vec3(1,0,0), Vec3(1,0,0)};
    std::vector<Vec3> colors  ={Vec3(0,0,0), Vec3(0,1,0), Vec3(1,1,0), Vec3(1,0,0)};
    uploadMeshData(vertices, indices, normals, colors);*/

}

CustomGeometry::~CustomGeometry()
{
}

void CustomGeometry::setRenderPoints(bool pointsNotTriangles)
{
    if (pointsNotTriangles)  setPrimitiveType(Qt3DRender::QGeometryRenderer::Points);
    else                     setPrimitiveType(Qt3DRender::QGeometryRenderer::Triangles);
}

void CustomGeometry::setRenderLines(bool linesNotTriangles)
{
    if (linesNotTriangles)  setPrimitiveType(Qt3DRender::QGeometryRenderer::Lines);
    else                    setPrimitiveType(Qt3DRender::QGeometryRenderer::Triangles);
}

void CustomGeometry::setEmptyData()
{
    std::vector<QVector3D> vertices;
    std::vector<int> indices;
  //  std::vector<QVector3D> normals;
   // std::vector<QVector3D> colors;
    std::vector<QVector2D> uvs;
    uploadMeshData(vertices, indices,uvs);
}

void CustomGeometry::uploadMeshData(std::vector<QVector3D>& vertices, std::vector<int>& indices,std::vector<QVector2D>& uvs)
{
    // vec3 for position
    // vec3 for colors
    // vec3 for normals

    size_t numVertices = vertices.size();
    size_t numIndices  = indices.size();

    QVector<QVector3D> verts;
    for (size_t i = 0; i <numVertices; i++)
        verts << vertices[i];// << normals[i] << colors[i];


    QByteArray vertexBufferData;
    vertexBufferData.resize(uint(numVertices) * m_valuesPerVertex * sizeof(float));
    float *rawVertexArray = reinterpret_cast<float *>(vertexBufferData.data());

    QByteArray texBufferData;
    texBufferData.resize(2*uint(numVertices) * sizeof(float));
    float* rawTexArray = reinterpret_cast<float*>(texBufferData.data());


    uint ct = 0;
    for (const QVector3D &v : verts)
    {
        rawVertexArray[ct++] = v.x();
        rawVertexArray[ct++] = v.y();
        rawVertexArray[ct++] = v.z();
    }
    ct=0;
    for (const QVector2D &v : uvs)
	{
    	rawTexArray[ct++] = v.x();
    	rawTexArray[ct++] = v.y();
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

    m_texBuffer->setData(texBufferData);
    m_positionAttribute->setCount(uint(numVertices));
   // m_normalAttribute->setCount(uint(numVertices));
  //   m_colorAttribute->setCount(uint(numVertices));
     m_textureAttribute->setCount(uint(numVertices));

 //   setVertexCount(numIndices);
}




//=====================================================================================================
RandomView3D::RandomView3D(WorkingSetManager* workingset, RandomLineView* random,GraphEditor_LineShape* line, QString nameView/*,Qt3DRender::QLayer* layer*/, Qt3DCore::QNode *parent) : Qt3DCore::QEntity(parent)
{
	m_selected = false;
	m_nameView = nameView;
	m_ratioSize = 1.0f;
	m_random = random;
	m_lineShape = line;
	m_workingset = workingset;


	connect(m_random,SIGNAL(destroyed()),this,SLOT( deleteRamdomLine()));

	/*m_randomData = new RandomDataset(workingset,this,m_nameView,this);

	QList<QString> listename = workingset->getDataset(m_randomData);


	for(int i=0;i<listename.count();i++)
	{
		RandomTexDataset* randomtex = new RandomTexDataset(m_workingset,listename.at(i),cudaTextures[i],ranges[0],this);
		m_randomData->addDataset(randomtex);
	}

	m_workingset->addRandom(m_randomData);
	m_randomData->setDisplayPreference(true);*/

	m_planeEntity = new Qt3DCore::QEntity(parent);

	m_currentMesh = new CustomGeometry( parent);
	m_transfo = new Qt3DCore::QTransform();



	m_planeEntity->addComponent(m_currentMesh);
	m_planeEntity->addComponent(m_transfo);
	//m_planeEntity->addComponent(layer);




	  // picker
	Qt3DRender::QObjectPicker* spicker = new Qt3DRender::QObjectPicker();
	Qt3DRender::QPickingSettings *pickingSettings = new Qt3DRender::QPickingSettings(spicker);
	pickingSettings->setPickMethod(Qt3DRender::QPickingSettings::TrianglePicking);
	pickingSettings->setPickResultMode(Qt3DRender::QPickingSettings::NearestPick);
	pickingSettings->setFaceOrientationPickingMode(Qt3DRender::QPickingSettings::FrontAndBackFace);
	pickingSettings->setEnabled(true);
	spicker->setEnabled(true);
	spicker->setDragEnabled(true);
	m_planeEntity->addComponent(spicker);//m_ghostEntity




	connect(spicker, &Qt3DRender::QObjectPicker::pressed, [&](Qt3DRender::QPickEvent* e) {
		m_actifRayon= true;
		});
		connect(spicker, &Qt3DRender::QObjectPicker::moved, [&](Qt3DRender::QPickEvent* e) {
		m_actifRayon= false;
		});


	connect(spicker, &Qt3DRender::QObjectPicker::clicked, [&](Qt3DRender::QPickEvent* e) {

	if(m_actifRayon== true)// && e->button() == Qt3DRender::QPickEvent::Buttons::LeftButton)
	{
		int bouton = e->button();
		auto p = dynamic_cast<Qt3DRender::QPickTriangleEvent*>(e);
		if(p) {
			QVector3D pos = p->worldIntersection();


			//emit sendPositionTarget(newpos,pos);

		emit sendAnimationCam(bouton,pos);
		}
	}
	});


}


RandomView3D::RandomView3D(WorkingSetManager* workingset, RandomLineView* random,GraphEditor_LineShape* line, QString nameView,
		QVector<CudaImageTexture*> cudaTextures,QVector<QVector2D> ranges,/*Qt3DRender::QLayer* layer,*/ Qt3DCore::QNode *parent) : Qt3DCore::QEntity(parent)
{
	m_selected = false;
	m_nameView = nameView;
	m_ratioSize = 1.0f;
	m_random = random;
	m_lineShape = line;
	m_workingset = workingset;

	connect(m_random,SIGNAL(destroyed()),this,SLOT( deleteRamdomLine()));

	m_randomData = new RandomDataset(workingset,this,m_nameView,this);

	QList<QString> listename = workingset->getDataset(m_randomData);


	for(int i=0;i<listename.count();i++)
	{
		RandomTexDataset* randomtex = new RandomTexDataset(m_workingset,listename.at(i),cudaTextures[i],ranges[i],this);
		m_randomData->addDataset(randomtex);
	}

	m_workingset->addRandom(m_randomData);
	m_randomData->setAllDisplayPreference(true);

	m_planeEntity = new Qt3DCore::QEntity(parent);

	m_currentMesh = new CustomGeometry( parent);
	m_transfo = new Qt3DCore::QTransform();



	m_planeEntity->addComponent(m_currentMesh);
	m_planeEntity->addComponent(m_transfo);
	//m_planeEntity->addComponent(layer);




	  // picker
	Qt3DRender::QObjectPicker* spicker = new Qt3DRender::QObjectPicker();
	Qt3DRender::QPickingSettings *pickingSettings = new Qt3DRender::QPickingSettings(spicker);
	pickingSettings->setPickMethod(Qt3DRender::QPickingSettings::TrianglePicking);
	pickingSettings->setPickResultMode(Qt3DRender::QPickingSettings::NearestPick);
	pickingSettings->setFaceOrientationPickingMode(Qt3DRender::QPickingSettings::FrontAndBackFace);
	pickingSettings->setEnabled(true);
	spicker->setEnabled(true);
	spicker->setDragEnabled(true);
	m_planeEntity->addComponent(spicker);//m_ghostEntity


	connect(spicker, &Qt3DRender::QObjectPicker::pressed, [&](Qt3DRender::QPickEvent* e) {
		m_actifRayon= true;
		});
		connect(spicker, &Qt3DRender::QObjectPicker::moved, [&](Qt3DRender::QPickEvent* e) {
		m_actifRayon= false;
		});


	connect(spicker, &Qt3DRender::QObjectPicker::clicked, [&](Qt3DRender::QPickEvent* e) {

	if(m_actifRayon== true)// && e->button() == Qt3DRender::QPickEvent::Buttons::LeftButton)
	{
		int bouton = e->button();
		auto p = dynamic_cast<Qt3DRender::QPickTriangleEvent*>(e);
		if(p) {
			QVector3D pos = p->worldIntersection();


			//emit sendPositionTarget(newpos,pos);

		emit sendAnimationCam(bouton,pos);
		}
	}
	});
}

void RandomView3D::deleteRamdomLine()
{
	if(m_random != nullptr)
	{
		m_random->close();
	}
	m_random=nullptr;
	emit destroy(this);


}

void RandomView3D::setParam(float width, float height)
{
	m_width= width;
	m_height= height;
}

void RandomView3D::refreshWidth(QVector<QVector3D> listepoints,float width)
{
	m_listePts = listepoints;
	m_width= width;
	m_currentMesh->setEmptyData();
	 std::vector<QVector3D> vertices;//,normals,colors;
	 std::vector<QVector2D> uvs;
	 std::vector<int> indices;

	 float widthTmp=0.0f;


	 float nbSection = m_listePts.count()-1;
	 for(int i=0;i<m_listePts.count();i++)
	 {

		 QVector3D pts1 =  m_listePts[i];
		 vertices.push_back(pts1);
		 QVector3D pts2 =  pts1+ QVector3D(0.0f,m_height,0.0f);
		 vertices.push_back(pts2);

		 uvs.push_back(QVector2D((float)i/nbSection,0.0f));
		 uvs.push_back(QVector2D((float)i/nbSection,1.0f));
		if( i>0)
		{
			widthTmp += (m_listePts[i]- m_listePts[i-1]).length();
		}

	 }

	 m_ratioSize= m_height/m_width;
	 for( int i=0;i<m_listePts.count()-1;i++)
	 {
		 int index = i*2;
		 indices.push_back(index);
		 indices.push_back(index+1);
		 indices.push_back(index+2);

		 indices.push_back(index+2);
		 indices.push_back(index+1);
		 indices.push_back(index+3);
	 }

	 m_currentMesh->uploadMeshData(vertices, indices,uvs);
}

void RandomView3D::init(QVector<QVector3D> listepoints,float width, float height) //,CudaImageTexture* cudaTexture,QVector2D range)
{
	m_listePts = listepoints;
	m_width= width;
	m_height= height;
/*
	 Qt3DRender::QTexture2D *      backgroundTexture  = new Qt3DRender::QTexture2D(m_material);
		Qt3DRender::QTextureImage *   backgroundImage    = new Qt3DRender::QTextureImage(m_material);
		backgroundImage->setSource(QUrl::fromLocalFile("Diffuse2.png"));
		backgroundTexture->addTextureImage(backgroundImage);
		m_material->setTexture(backgroundTexture);*/
	//m_material->setTexture(cudaTexture);

	 m_currentMesh->setEmptyData();
	 std::vector<QVector3D> vertices;//,normals,colors;
	 std::vector<QVector2D> uvs;
	 std::vector<int> indices;

	 float widthTmp=0.0f;


	 float nbSection = listepoints.count()-1;
	 for(int i=0;i<listepoints.count();i++)
	 {

		 QVector3D pts1 =  listepoints[i];
		 vertices.push_back(pts1);
		 QVector3D pts2 =  pts1+ QVector3D(0.0f,height,0.0f);
		 vertices.push_back(pts2);

		 uvs.push_back(QVector2D((float)i/nbSection,0.0f));
		 uvs.push_back(QVector2D((float)i/nbSection,1.0f));
		if( i>0)
		{
			widthTmp += (listepoints[i]- listepoints[i-1]).length();
		}

	 }

	 m_ratioSize= height/width;
	 for( int i=0;i<listepoints.count()-1;i++)
	 {
		 int index = i*2;
		 indices.push_back(index);
		 indices.push_back(index+1);
		 indices.push_back(index+2);

		 indices.push_back(index+2);
		 indices.push_back(index+1);
		 indices.push_back(index+3);
	 }

	 m_currentMesh->uploadMeshData(vertices, indices,uvs);

	//initMaterial(cudaTexture,range);
}


void RandomView3D::update(QVector3D pos, QVector3D normal)
{

	if( normal.length()< 0.25f)
	{
	//	qDebug()<<"RandomView3D::update => normal == zero ";
		return;
	}
	float Y = m_listePts[0].y();


	QVector3D up(0.0,-1.0f,0.0f) ;
	QVector3D right = QVector3D::crossProduct(normal, up);
	right = right.normalized();

	QVector3D posHG = pos - right * m_width*0.5+ up *m_height*0.5 ; //(pos.x()-0.5f * m_width, pos.y()-0.5f * m_height)
	QVector3D posHD = pos + right * m_width*0.5+ up *m_height*0.5;

	posHG.setY(Y);
	posHD.setY(Y);

	//QVector3D posBG = pos - right * m_width*0.5- up *m_height; //(pos.x()-0.5f * m_width, pos.y()-0.5f * m_height)
	//QVector3D posBD = pos + right * m_width*0.5- up *m_height;

	QVector<QVector3D> listepoints;
	listepoints.push_back(posHG);
	listepoints.push_back(posHD);

	init(listepoints,m_width,m_height);


}

void RandomView3D::initMaterial(CudaImageTexture* cudaTexture,QVector2D range)
{


	// Set the effect on the materials
	m_material = new Qt3DRender::QMaterial();
	m_material->setEffect(Qt3DHelpers::generateImageEffect("qrc:/shaders/qt3d/simpleTex.frag","qrc:/shaders/qt3d/simpleTex.vert"));

	m_parameterTexture = new Qt3DRender::QParameter(QStringLiteral("redMap"),cudaTexture);
	m_material->addParameter(m_parameterTexture);

	m_parameterRange = new Qt3DRender::QParameter(QStringLiteral("redRange"),range);
	m_material->addParameter(m_parameterRange);

	m_parameterHover = new Qt3DRender::QParameter(QStringLiteral("hover"),m_selected);
	m_material->addParameter(m_parameterHover);

	m_material->addParameter(new Qt3DRender::QParameter(QStringLiteral("ratio"),m_ratioSize));


	m_planeEntity->addComponent(m_material);
}

void RandomView3D::updateMaterial(CudaImageTexture* cudaTexture,QVector2D range)
{
	if(m_parameterTexture != nullptr) m_parameterTexture->setValue(QVariant::fromValue(cudaTexture));
	if(m_parameterRange != nullptr) m_parameterRange->setValue(range);

	/*QVector<Qt3DRender::QParameter *> params = m_material->parameters();
	params[0] = new Qt3DRender::QParameter(QStringLiteral("redMap"),cudaTexture);
	params[1] = new Qt3DRender::QParameter(QStringLiteral("redRange"),range);
*/
	//m_material->parameters = params;
	//m_material->parameters[0]->setValue(cudaTexture);
	//m_material->parameters[1]->setValue(range);
}

void RandomView3D::setZScale(float sca)
{
	m_transfo->setScale3D(QVector3D(1.0f,sca,1.0f));
}

bool RandomView3D::isEquals(QString name)
{
	return (m_nameView== name);
}

bool RandomView3D::isEquals(QVector<QVector3D> listepoints)
{
	if(m_listePts.count() != listepoints.count())
		return false;

	for(int i=0;m_listePts.count();i++)
	{
		bool res = qFuzzyCompare(m_listePts[i], listepoints[i]);
		if(res==false) return false;
	}

	return true;
}

void RandomView3D::setSelected(bool b)
{
	m_selected = b;
	if(m_parameterHover != nullptr) m_parameterHover->setValue(m_selected);
}

QVector<QVector3D> RandomView3D::getPoints()
{
	return m_listePts;
}

QString RandomView3D::getName()
{
	return m_nameView;
}

RandomView3D::~RandomView3D()
{
	if(m_planeEntity!= nullptr)
	{
		delete m_planeEntity;
		m_planeEntity = nullptr;
	}
	if(m_currentMesh!= nullptr)
	{
		delete m_currentMesh;
		m_currentMesh = nullptr;
	}

	if(m_randomData != nullptr)
	{
		m_workingset->removeRandom(m_randomData);
	}
/*	if(m_lineShape!= nullptr)
	{
		delete m_lineShape;
		m_lineShape = nullptr;
	}*/
	m_random = nullptr;
	m_lineShape = nullptr;


}

void RandomView3D::setColorCross(QColor c)
{
   	if(m_random)
   	{
   		m_random->setColorCross(c);
   	}
}

RandomLineView* RandomView3D::getRandomLineView()
{
	return m_random;
}

void RandomView3D::destroyRandom()
{
	if(m_random != nullptr)
	{

		m_random->close();
		//delete m_random;
		//m_random->deleteLater();
		//m_random = nullptr;
	}
}

void RandomView3D::show()
{
	if(m_planeEntity != nullptr)m_planeEntity->setEnabled(true);
}

void RandomView3D::hide()
{
	if(m_planeEntity != nullptr) m_planeEntity->setEnabled(false);
}





