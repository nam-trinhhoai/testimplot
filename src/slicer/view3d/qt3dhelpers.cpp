#include "qt3dhelpers.h"
#include <Qt3DRender/QTechnique>
#include <QEffect>
#include <QGraphicsApiFilter>
#include <QShaderProgram>
#include <Qt3DRender/QCullFace>
#include <Qt3DRender/QDepthTest>
#include <Qt3DRender/QNoDepthMask>

#include <Qt3DRender/QBlendEquationArguments>
#include <Qt3DRender/QBlendEquation>
#include <Qt3DRender/QLineWidth>
#include <Qt3DCore/QEntity>
#include <QCamera>
#include <QWindow>
#include <QUrl>
#include <QRandomGenerator>
#include <QQuickItem>
#include <qmath.h>

#include <Qt3DCore/QTransform>
#include <Qt3DExtras/QPhongMaterial>
#include <Qt3DExtras/QPerVertexColorMaterial>
#include <Qt3DCore/QAttribute>
#include <Qt3DCore/QBuffer>
#include <Qt3DCore/QGeometry>
#include <Qt3DRender/QGeometryRenderer>
#include <Qt3DRender/QPickTriangleEvent>

/*
void Qt3DHelpers::loadInp(Qstring s)
{
	QVector<QVector3D> positions;
	QVector<int> indices;

	0 x0, y0 ,z0
	1 x1, y1 ,z1
	2 x2, y2 ,z2
	3 x3, y3 ,z3


	0 1 2 0 2 3

}*/

Qt3DCore::QEntity* renderMesh(Qt3DCore::QEntity* _rootEntity,QVector<QVector3D> nodes,QVector<int> nodeIndexes,int nTriangles)
{
	int nNodes = nodes.count();

	QByteArray bufferBytes;
			bufferBytes.resize(3 * nNodes * sizeof(float)); // start.x, start.y, start.end + end.x, end.y, end.z
			float *positions = reinterpret_cast<float*>(bufferBytes.data());

			QByteArray indexBytes;
			indexBytes.resize(3*nTriangles * sizeof(unsigned int)); // start to end
			unsigned int *indices = reinterpret_cast<unsigned int*>(indexBytes.data());

		/*	QByteArray uvBytes;
			uvBytes.resize(2*nNodes * sizeof(float));
			float* uvtex = reinterpret_cast<float*>(uvBytes.data());


			  QByteArray normalBytes;
			normalBytes.resize(3 * nNodes * sizeof(float));
			float *normals = reinterpret_cast<float*>(normalBytes.data());
	*/

		auto *geometry = new Qt3DCore::QGeometry(_rootEntity);

			auto *buf = new Qt3DCore::QBuffer(geometry);
			buf->setData(bufferBytes);

			auto *positionAttribute = new Qt3DCore::QAttribute(geometry);
			positionAttribute->setName(Qt3DCore::QAttribute::defaultPositionAttributeName());
			positionAttribute->setVertexBaseType(Qt3DCore::QAttribute::Float);
			positionAttribute->setVertexSize(3);
			positionAttribute->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
			positionAttribute->setBuffer(buf);
			positionAttribute->setByteStride(3 * sizeof(float));
			positionAttribute->setCount(nNodes);
			geometry->addAttribute(positionAttribute); // We add the vertices in the geometry


			auto *indexBuffer = new Qt3DCore::QBuffer(geometry);
			indexBuffer->setData(indexBytes);

			auto *indexAttribute = new Qt3DCore::QAttribute(geometry);
			indexAttribute->setVertexBaseType(Qt3DCore::QAttribute::UnsignedInt);
			indexAttribute->setAttributeType(Qt3DCore::QAttribute::IndexAttribute);
			indexAttribute->setBuffer(indexBuffer);
			indexAttribute->setCount(nTriangles*3);
			geometry->addAttribute(indexAttribute); // We add the indices linking the points in the geometry

		/*	auto *normalBuffer = new Qt3DRender::QBuffer(geometry);
			normalBuffer->setData(normalBytes);

			auto *normalAttribute = new Qt3DRender::QAttribute(geometry);
			normalAttribute->setName(Qt3DRender::QAttribute::defaultNormalAttributeName());
			normalAttribute->setVertexBaseType(Qt3DRender::QAttribute::Float);
			normalAttribute->setVertexSize(3);
			normalAttribute->setAttributeType(Qt3DRender::QAttribute::VertexAttribute);
			normalAttribute->setBuffer(normalBuffer);
			normalAttribute->setByteStride(3 * sizeof(float));
			normalAttribute->setCount(nbvertex);
			geometry->addAttribute(normalAttribute);


		    Qt3DRender::QAttribute *m_textureAttribute;

		    auto *texBuffer = new Qt3DRender::QBuffer(geometry);
		    auto *texAttribute = new Qt3DRender::QAttribute(geometry);

		    texBuffer->setData(uvBytes);
		    texBuffer->setType(Qt3DRender::QBuffer::VertexBuffer);
		    texAttribute->setName(Qt3DRender::QAttribute::defaultTextureCoordinateAttributeName());
		    texAttribute->setAttributeType(Qt3DRender::QAttribute::VertexAttribute);
		    texAttribute->setDataType(Qt3DRender::QAttribute::Float);
		    texAttribute->setBuffer(texBuffer);
		    texAttribute->setDataSize(2);
		    texAttribute->setByteOffset(0);
		    texAttribute->setByteStride(0);
		    texAttribute->setCount(nbvertex);

			 geometry->addAttribute(texAttribute);*/



			auto *objet = new Qt3DRender::QGeometryRenderer(_rootEntity);
			objet->setGeometry(geometry);
			objet->setPrimitiveType(Qt3DRender::QGeometryRenderer::Triangles);   //Triangles);


		   auto *objEntity = new Qt3DCore::QEntity(_rootEntity);
		   objEntity->addComponent(objet);

		   return objEntity;
}
/*
 void Qt3DHelpers::view_generated_mesh()
{
	QString const fileName = QFileDialog::getOpenFileName(this,
		tr("Load mesh file"), lagrit_path.c_str(), tr("Mesh file (*.inp)"));

	QFile file(fileName);
	file.open(QFile::ReadOnly);

	QTextStream stream(&file);

	QStringList const firstLineList = stream.readLine().split(" ", QString::SkipEmptyParts);

	int const nNodes = firstLineList[0].toInt();
	int const nElements = firstLineList[1].toInt();

	QVector<QVector3D> nodes;

	// Get nodes coordinates
	nodes.resize(nNodes);

	for(int i=0; i<nNodes; i++)
	{
		QStringList const lineList = stream.readLine().split(" ", QString::SkipEmptyParts);

		nodes[i] = QVector3D(lineList[1].toDouble(),
							 lineList[2].toDouble(),
							 lineList[3].toDouble());
	}

	// Get faces triangles node indexes

	int nTriangles = nElements*6*2*3; // Each hex element has 6 faces,each face is devided into two triangles, each triangle has three nodes

	QVector<int> nodeIndexes_for_faces_triangles;
	nodeIndexes_for_faces_triangles.resize(nTriangles);
	int idx = 0;

	for(int i=0; i<nElements; i++)
	{
		QStringList lineList = stream.readLine().split(" ", QString::SkipEmptyParts);

		// Get the node indexes of each Hex element
		QVector<int> nodeIndexes(8);
		for(int j=0; j<8; j++)
		{
			nodeIndexes[j] = lineList[j+3].toInt();
		}

		// Add element nodes to the liste of face nodes
		// Top face
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[0];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[1];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[2];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[0];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[2];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[3];

		// Bottom face
		nodeIndexes_for_faces_triangles[idx++]  = nodeIndexes[4];
		nodeIndexes_for_faces_triangles[idx++]  = nodeIndexes[7];
		nodeIndexes_for_faces_triangles[idx++]  = nodeIndexes[6];
		nodeIndexes_for_faces_triangles[idx++]  = nodeIndexes[4];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[6];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[5];

		// Left face
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[0];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[3];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[7];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[0];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[7];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[4];

		// Right face
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[1];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[5];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[6];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[1];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[6];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[2];

		// Front face
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[0];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[4];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[5];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[0];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[5];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[1];

		// Back face
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[2];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[6];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[7];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[2];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[7];
		nodeIndexes_for_faces_triangles[idx++] = nodeIndexes[3];
	}


//	for(int i=0; i<nodeIndexes_for_faces_triangles.count(); i++)
//	{

//		std::cout << nodeIndexes_for_faces_triangles[i] << std::endl;
//	}


	file.close();


}
*/

Qt3DRender::QEffect * Qt3DHelpers::generateImageEffect(const QString& fragPath, const QString & vertPath)
{

	//qDebug()<<vertPath<<" , generateImageEffect "<<fragPath;
	Qt3DRender::QEffect *effect = new Qt3DRender::QEffect();
	Qt3DRender::QTechnique *gl3Technique = new Qt3DRender::QTechnique();
	Qt3DRender::QRenderPass *gl3Pass = new Qt3DRender::QRenderPass();

	Qt3DRender::QFilterKey *filter = new Qt3DRender::QFilterKey();
	filter->setName(QStringLiteral("renderingStyle"));
	filter->setValue(QStringLiteral("forward"));
	gl3Technique->addFilterKey(filter);

	// Set the targeted GL version for the technique
	gl3Technique->graphicsApiFilter()->setApi(
			Qt3DRender::QGraphicsApiFilter::OpenGL);
	gl3Technique->graphicsApiFilter()->setMajorVersion(3);
	gl3Technique->graphicsApiFilter()->setMinorVersion(1);
	gl3Technique->graphicsApiFilter()->setProfile(
			Qt3DRender::QGraphicsApiFilter::CoreProfile);

	Qt3DRender::QShaderProgram *glShader = new Qt3DRender::QShaderProgram();

	QByteArray fragArray = Qt3DRender::QShaderProgram::loadSource(QUrl(fragPath));


	glShader->setFragmentShaderCode(fragArray);


	QByteArray vertArray = Qt3DRender::QShaderProgram::loadSource(QUrl(vertPath));
	glShader->setVertexShaderCode(vertArray);
	//qDebug()<<"vertex ==========>" <<vertArray;
	gl3Pass->setShaderProgram(glShader);

	connect(glShader,&Qt3DRender::QShaderProgram::statusChanged,[glShader,vertPath,fragPath](Qt3DRender::QShaderProgram::Status status)
	{
		//qDebug()<<"======> glShader::statusChanged"<<status;
		if (glShader->status() != Qt3DRender::QShaderProgram::Ready)
		{
		//	if (glShader->status() == Qt3DRender::QShaderProgram::NotReady) qDebug("->glShader::NotReady");
			if (glShader->status() == Qt3DRender::QShaderProgram::Error)    qDebug() << vertPath<<" , "<< fragPath<<" , glShader::Error" << glShader->log();
		}
		//else qDebug()<<" READY ==>"<< vertPath<<" , "<< fragPath;
	});


	 if (glShader->status() != Qt3DRender::QShaderProgram::Ready)    {
	            //  if (glShader->status() == Qt3DRender::QShaderProgram::NotReady) qDebug("-->glShader::NotReady");
	              if (glShader->status() == Qt3DRender::QShaderProgram::Error)    qDebug() << "glShader::Error" << glShader->log();
	        } //else qDebug("======> glShader::Ready");

	Qt3DRender::QCullFace *cull = new Qt3DRender::QCullFace();
	cull->setMode(Qt3DRender::QCullFace::NoCulling);
	gl3Pass->addRenderState(cull);

	Qt3DRender::QDepthTest *depthTest = new Qt3DRender::QDepthTest();
	depthTest->setDepthFunction(Qt3DRender::QDepthTest::LessOrEqual);
	gl3Pass->addRenderState(depthTest);



	/*if(fragPath =="RGBColor_simple.frag")
	{
		Qt3DRender::QDepthTest *depthTest = new Qt3DRender::QDepthTest();
			depthTest->setDepthFunction(Qt3DRender::QDepthTest::Never);
			gl3Pass->addRenderState(depthTest);
	}
	else
	{
		Qt3DRender::QDepthTest *depthTest = new Qt3DRender::QDepthTest();
			depthTest->setDepthFunction(Qt3DRender::QDepthTest::LessOrEqual);
			gl3Pass->addRenderState(depthTest);
	}*/





	if(fragPath.contains("rgbPhong.frag"))
	{

			//qDebug()<<" shader rgbPhong...";
			Qt3DRender::QBlendEquationArguments *blendEq =new Qt3DRender::QBlendEquationArguments();


			blendEq->setSourceAlpha(Qt3DRender::QBlendEquationArguments::SourceColor);//SourceAlpha
			blendEq->setDestinationAlpha(Qt3DRender::QBlendEquationArguments::OneMinusSourceAlpha);//OneMinusSourceAlpha
			blendEq->setSourceRgb(Qt3DRender::QBlendEquationArguments::SourceAlpha);//SourceAlpha
			blendEq->setDestinationRgb(Qt3DRender::QBlendEquationArguments::OneMinusSourceAlpha);//OneMinusSourceAlpha

			Qt3DRender::QDepthTest *depthTest = new Qt3DRender::QDepthTest();
			depthTest->setDepthFunction(Qt3DRender::QDepthTest::LessOrEqual);
			gl3Pass->addRenderState(depthTest);



		//	Qt3DRender::QNoDepthMask *nodepthmask = new Qt3DRender::QNoDepthMask();
		//	gl3Pass->addRenderState(nodepthmask);

			gl3Pass->addRenderState(blendEq);

			Qt3DRender::QBlendEquation *blend = new Qt3DRender::QBlendEquation();
			blend->setBlendFunction(Qt3DRender::QBlendEquation::Add);
			gl3Pass->addRenderState(blend);
		}



	gl3Technique->addRenderPass(gl3Pass);

	// Add the technique to the effect
	effect->addTechnique(gl3Technique);
	return effect;
}


QVector3D Qt3DHelpers::computeCameraRay(QWindow *parent,Qt3DRender::QCamera *camera,
		const QVector2D &position) {
	QMatrix4x4 viewMatrix;
	viewMatrix.lookAt(camera->position(), camera->viewCenter(),
			camera->upVector());

	double x = ((2.0 * position.x()) / parent->width()) - 1.0;
	double y = 1.0 - ((2.0 * position.y()) / parent->height());

	QVector4D ray = camera->projectionMatrix().inverted()
			* QVector4D(x, y, -1.0, 1.0);
	ray.setZ(-1.0);
	ray.setW(0);
	ray = viewMatrix.inverted() * ray;
	return ray.toVector3D().normalized();
}

bool Qt3DHelpers::intersectPlane(QVector3D &result,
		const QVector3D &planePoint, const QVector3D &planeNormal,
		const QVector3D &linePoint, const QVector3D &lineDirection) {
	if (QVector3D::dotProduct(planeNormal, lineDirection) == 0)
		return false;

	float t = (QVector3D::dotProduct(planeNormal, planePoint)
			- QVector3D::dotProduct(planeNormal, linePoint))
			/ QVector3D::dotProduct(planeNormal, lineDirection);
	QVector3D t1 = lineDirection * t;
	result = linePoint + t1;
	return true;
}

Qt3DCore::QEntity* Qt3DHelpers::drawLine(const QVector3D& start, const QVector3D& end, const QColor& color, Qt3DCore::QEntity *_rootEntity)
{
	auto *geometry = new Qt3DCore::QGeometry(_rootEntity);

	// position vertices (start and end)
	QByteArray bufferBytes;
	bufferBytes.resize(3 * 2 * sizeof(float)); // start.x, start.y, start.end + end.x, end.y, end.z
	float *positions = reinterpret_cast<float*>(bufferBytes.data());
	*positions++ = start.x();
	*positions++ = start.y();
	*positions++ = start.z();
	*positions++ = end.x();
	*positions++ = end.y();
	*positions++ = end.z();

	auto *buf = new Qt3DCore::QBuffer(geometry);
	buf->setData(bufferBytes);

	auto *positionAttribute = new Qt3DCore::QAttribute(geometry);
	positionAttribute->setName(Qt3DCore::QAttribute::defaultPositionAttributeName());
	positionAttribute->setVertexBaseType(Qt3DCore::QAttribute::Float);
	positionAttribute->setVertexSize(3);
	positionAttribute->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
	positionAttribute->setBuffer(buf);
	positionAttribute->setByteStride(3 * sizeof(float));
	positionAttribute->setCount(2);
	geometry->addAttribute(positionAttribute); // We add the vertices in the geometry

	// connectivity between vertices
	QByteArray indexBytes;
	indexBytes.resize(2 * sizeof(unsigned int)); // start to end
	unsigned int *indices = reinterpret_cast<unsigned int*>(indexBytes.data());
	*indices++ = 0;
	*indices++ = 1;

	auto *indexBuffer = new Qt3DCore::QBuffer(geometry);
	indexBuffer->setData(indexBytes);

	auto *indexAttribute = new Qt3DCore::QAttribute(geometry);
	indexAttribute->setVertexBaseType(Qt3DCore::QAttribute::UnsignedInt);
	indexAttribute->setAttributeType(Qt3DCore::QAttribute::IndexAttribute);
	indexAttribute->setBuffer(indexBuffer);
	indexAttribute->setCount(2);
	geometry->addAttribute(indexAttribute); // We add the indices linking the points in the geometry

	// mesh
	auto *line = new Qt3DRender::QGeometryRenderer(_rootEntity);
	line->setGeometry(geometry);
	line->setPrimitiveType(Qt3DRender::QGeometryRenderer::Lines);


	auto *material = new Qt3DExtras::QPhongMaterial(_rootEntity);
	material->setAmbient(color);


	// entity
	auto *lineEntity = new Qt3DCore::QEntity(_rootEntity);
	lineEntity->addComponent(line);
	lineEntity->addComponent(material);


	return lineEntity;
}

Qt3DCore::QEntity* Qt3DHelpers::drawNormals(const QVector<QVector3D>& positionsVec,const QColor& color, Qt3DCore::QEntity *_rootEntity, int lineWidth)
{


	auto *geometry = new Qt3DCore::QGeometry(_rootEntity);

	int nbVertex = positionsVec.size();
	QByteArray bufferBytes;
	bufferBytes.resize(3 * nbVertex * sizeof(float)); // start.x, start.y, start.end + end.x, end.y, end.z
	float *positions = reinterpret_cast<float*>(bufferBytes.data());


	for(int i=0;i<nbVertex;i++)
	{
		*positions++ = positionsVec[i].x();
		*positions++ = positionsVec[i].y();
		*positions++ = positionsVec[i].z();

	}


	auto *buf = new Qt3DCore::QBuffer(geometry);
	buf->setData(bufferBytes);

	auto *positionAttribute = new Qt3DCore::QAttribute(geometry);
	positionAttribute->setName(Qt3DCore::QAttribute::defaultPositionAttributeName());
	positionAttribute->setVertexBaseType(Qt3DCore::QAttribute::Float);
	positionAttribute->setVertexSize(3);
	positionAttribute->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
	positionAttribute->setBuffer(buf);
	positionAttribute->setByteStride(3 * sizeof(float));
	positionAttribute->setCount(nbVertex);
	geometry->addAttribute(positionAttribute); // We add the vertices in the geometry

	// connectivity between vertices
	QByteArray indexBytes;
	indexBytes.resize(nbVertex * sizeof(unsigned int)); // start to end
	unsigned int *indices = reinterpret_cast<unsigned int*>(indexBytes.data());
	for(int i=0;i<nbVertex;i++)
	{
		*indices++ = i;

	}
//	*indices++ = 0;
	///*indices++ = 1;

	auto *indexBuffer = new Qt3DCore::QBuffer(geometry);
	indexBuffer->setData(indexBytes);

	auto *indexAttribute = new Qt3DCore::QAttribute(geometry);
	indexAttribute->setVertexBaseType(Qt3DCore::QAttribute::UnsignedInt);
	indexAttribute->setAttributeType(Qt3DCore::QAttribute::IndexAttribute);
	indexAttribute->setBuffer(indexBuffer);
	indexAttribute->setCount(nbVertex);
	geometry->addAttribute(indexAttribute); // We add the indices linking the points in the geometry

	// mesh
	auto *line = new Qt3DRender::QGeometryRenderer(_rootEntity);
	line->setGeometry(geometry);
	line->setPrimitiveType(Qt3DRender::QGeometryRenderer::Lines);

	auto *material = new Qt3DExtras::QPhongMaterial(_rootEntity);
	material->setAmbient(color);

	QVector<Qt3DRender::QTechnique *> tech = material->effect()->techniques();

	for(int i=0;i< tech.size();i++)
	{
		QVector<Qt3DRender::QRenderPass *> passes = tech[i]->renderPasses();
		for(int j=0;j< passes.size();j++)
		{
			Qt3DRender::QLineWidth* lineWith =new Qt3DRender::QLineWidth() ;
			lineWith->setValue(lineWidth);
			 passes[j]->addRenderState(lineWith);

		}
	}

	// entity
	auto *lineEntity = new Qt3DCore::QEntity(_rootEntity);
	lineEntity->addComponent(line);
	lineEntity->addComponent(material);

	return lineEntity;
}


Qt3DCore::QEntity* Qt3DHelpers::drawLines(const QVector<QVector3D>& positionsVec,const QColor& color, Qt3DCore::QEntity *_rootEntity, int lineWidth)
{
	auto *geometry = new Qt3DCore::QGeometry(_rootEntity);

	int nbVertex = positionsVec.size();
	QByteArray bufferBytes;
	bufferBytes.resize(3 * nbVertex * sizeof(float)); // start.x, start.y, start.end + end.x, end.y, end.z
	float *positions = reinterpret_cast<float*>(bufferBytes.data());


	for(int i=0;i<nbVertex;i++)
	{
		*positions++ = positionsVec[i].x();
		*positions++ = positionsVec[i].y();
		*positions++ = positionsVec[i].z();

	}


	auto *buf = new Qt3DCore::QBuffer(geometry);
	buf->setData(bufferBytes);

	auto *positionAttribute = new Qt3DCore::QAttribute(geometry);
	positionAttribute->setName(Qt3DCore::QAttribute::defaultPositionAttributeName());
	positionAttribute->setVertexBaseType(Qt3DCore::QAttribute::Float);
	positionAttribute->setVertexSize(3);
	positionAttribute->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
	positionAttribute->setBuffer(buf);
	positionAttribute->setByteStride(3 * sizeof(float));
	positionAttribute->setCount(nbVertex);
	geometry->addAttribute(positionAttribute); // We add the vertices in the geometry

	// connectivity between vertices
	QByteArray indexBytes;
	indexBytes.resize((2*nbVertex-2) * sizeof(unsigned int)); // start to end
	unsigned int *indices = reinterpret_cast<unsigned int*>(indexBytes.data());
	for(int i=0;i<nbVertex-1;i++)
	{
		*indices++ = i;
		*indices++ = i+1;
	}
//	*indices++ = 0;
	///*indices++ = 1;

	auto *indexBuffer = new Qt3DCore::QBuffer(geometry);
	indexBuffer->setData(indexBytes);

	auto *indexAttribute = new Qt3DCore::QAttribute(geometry);
	indexAttribute->setVertexBaseType(Qt3DCore::QAttribute::UnsignedInt);
	indexAttribute->setAttributeType(Qt3DCore::QAttribute::IndexAttribute);
	indexAttribute->setBuffer(indexBuffer);
	indexAttribute->setCount(2*nbVertex-2);
	geometry->addAttribute(indexAttribute); // We add the indices linking the points in the geometry

	// mesh
	auto *line = new Qt3DRender::QGeometryRenderer(_rootEntity);
	line->setGeometry(geometry);
	line->setPrimitiveType(Qt3DRender::QGeometryRenderer::Lines);

	auto *material = new Qt3DExtras::QPhongMaterial(_rootEntity);
	material->setAmbient(color);

	QVector<Qt3DRender::QTechnique *> tech = material->effect()->techniques();

	for(int i=0;i< tech.size();i++)
	{
		QVector<Qt3DRender::QRenderPass *> passes = tech[i]->renderPasses();
		for(int j=0;j< passes.size();j++)
		{
			Qt3DRender::QLineWidth* lineWith =new Qt3DRender::QLineWidth() ;
			lineWith->setValue(lineWidth);
			 passes[j]->addRenderState(lineWith);

		}
	}

	// entity
	auto *lineEntity = new Qt3DCore::QEntity(_rootEntity);
	lineEntity->addComponent(line);
	lineEntity->addComponent(material);

	return lineEntity;
}



QVector3D Qt3DHelpers::computePerpendicularVector(QVector3D vec)
{
    return vec.z() < vec.x() ? QVector3D(vec.y(), -vec.x(), 0) : QVector3D(0, -vec.z(), vec.y());
}

Qt3DRender::QCamera* Qt3DHelpers::sCamera = nullptr;
//Qt3DRender::QObjectPicker* Qt3DHelpers::sPicker = nullptr;
//Qt3DRender::QPickingSettings* Qt3DHelpers::sPickingSettings = nullptr;

qreal Qt3DHelpers::mix(qreal a0, qreal a1, qreal a2, qreal a3, qreal x)
{
    qreal p,q,r,s;
    p = (a3 - a2) - (a0 - a1);
    q = (a0 - a1) - p;
    r = a2 - a0;
    s = a1;

    return p*x*x*x + q*x*x + r*x + s;
}

//cubic interpolation between two 3D vectors
QVector3D Qt3DHelpers::mix(QVector3D p0, QVector3D p1, QVector3D p2, QVector3D p3, qreal x)
{
    return QVector3D(
                mix(p0.x(), p1.x(), p2.x(), p3.x(), x),
                mix(p0.y(), p1.y(), p2.y(), p3.y(), x),
                mix(p0.z(), p1.z(), p2.z(), p3.z(), x));
}

Qt3DCore::QEntity* Qt3DHelpers::drawLog(const QVector<QVector3D>& positionsVec,const QVector<float>& widthVec,const QColor& color, Qt3DCore::QEntity *_rootEntity, int lineWidth)
{

	if(positionsVec.size() < 2)
	    {
	        qWarning() << "Qt3DHelpers::drawExtrudersNew: insufficient number of input points.";
	        return new Qt3DCore::QEntity(_rootEntity);
	    }

	    int nbpoints = 0;

	   // qDebug()<<" position vec :"<<positionsVec.size();
	    auto *geometry = new Qt3DCore::QGeometry(_rootEntity);

	    int numInputPoints = positionsVec.size();
	    int resolution = 1;//30
	    int numRotations = 8;//30

	  //  QVector<QVector<QVector3D> > offsettPositionsVec;
	    QVector<QVector3D> resampledPositionsVec;
	   // QVector<QVector3D> resampledColorVec;
	    QVector<float> resampledWidthVec;


	  //  QVector<QVector3D> myPositionsVec;

	    for (int i = 0; i < (numInputPoints-1); ++i)
	    {
	        QVector3D aPos,bPos,cPos,dPos;
	        float aWidth,bWidth,cWidth,dWidth;


	        bPos = positionsVec[i];
	        cPos = positionsVec[i+1];

	        bWidth = widthVec[i];
	        cWidth = widthVec[i+1];

	        if(i > 0) {
	            aPos = positionsVec[i-1];
	            aWidth = widthVec[i-1];
	        } else {
	            aPos = positionsVec[i];
	            aWidth = widthVec[i];
	        }
	        if(i < numInputPoints - 2) {
	            dPos = positionsVec[i+2];
	            dWidth = widthVec[i+2];
	        } else {
	            dPos = positionsVec[i+1];
	            dWidth = widthVec[i+1];
	        }

	        // linear interpolation
	        if(i==0 || bWidth==cWidth)
	        {
	            for (int j = 0; j < resolution; ++j)
	            {
	                float value = (float)j / resolution;

	                QVector3D resampledPosition = (1.0-value) * bPos + value * cPos;
	                float width = (1.0-value) * bWidth + value * cWidth;

	                resampledPositionsVec.push_back(resampledPosition);
	                resampledWidthVec.push_back(width);
	              //  resampledColorVec.push_back(colorVec[i]);
	            }
	        }
	        // cubic interpolation
	        else
	        {
	            for (int j = 0; j < resolution; ++j)
	            {
	                float value = (float)j / resolution;

	                QVector3D resampledPosition = mix(aPos,bPos,cPos,dPos, value);
	                float width = mix(aWidth,bWidth,cWidth,dWidth, value);


	                resampledPositionsVec.push_back(resampledPosition);
	                resampledWidthVec.push_back(width);
	              //  resampledColorVec.push_back(colorVec[i]);

	            }
	        }
	    }

	    int numResampled = resampledPositionsVec.size();
	   QVector<QVector3D> resampledTangentVec;
	   QVector<QVector3D> resampledNormalVec;

	   for (int i = 0; i < numResampled; ++i)
	   {
		   QVector3D tangent;
		   if(i < (numResampled - 1)) {
			   tangent = (resampledPositionsVec[i+1] - resampledPositionsVec[i]).normalized();
		   } else
		   {
			   tangent = (resampledPositionsVec[i] - resampledPositionsVec[i-1]).normalized();
		   }

		   QVector3D normal = computePerpendicularVector(tangent).normalized();

		   resampledTangentVec.push_back(tangent);
		   resampledNormalVec.push_back(normal);
	   }

	   for (int i = 0; i < (numResampled-1); ++i)
	       {
	           QVector3D t_i = resampledTangentVec[i];
	           QVector3D t_i1 = resampledTangentVec[i+1];
	           QVector3D r_i = resampledNormalVec[i];

	           QVector3D v_1 = resampledPositionsVec[i + 1] - resampledPositionsVec[i];
	           float c1 = QVector3D::dotProduct(v_1, v_1);
	           QVector3D r_L_i = r_i - (2.0/c1) * QVector3D::dotProduct(v_1, r_i) * v_1;
	           QVector3D t_L_i = t_i - (2.0/c1) * QVector3D::dotProduct(v_1, t_i) * v_1;
	           QVector3D v_2  = t_i1 - t_L_i;
	           float c2 = QVector3D::dotProduct(v_2, v_2);
	           QVector3D r_i1 = r_L_i - (2.0/c2) * QVector3D::dotProduct(v_2, r_L_i) * v_2;

	           resampledNormalVec[i + 1] = r_i1;


	       }

	   QVector<QVector3D> myPositionVec;
	   for (int i = 0; i < (numResampled); ++i)
	   {
		   myPositionVec.push_back(resampledPositionsVec[i] + resampledNormalVec[i]*resampledWidthVec[i]);
	   	}


	    resampledPositionsVec.push_back(positionsVec.last());
	    resampledWidthVec.push_back(widthVec.last());
	 //   resampledColorVec.push_back(colorVec.last());

	   return  Qt3DHelpers::drawLines(myPositionVec,color,_rootEntity,lineWidth);



}
Qt3DCore::QEntity* Qt3DHelpers::drawExtruders(const QVector<QVector3D>& positionsVec, const QVector<QVector3D>& colorVec, const QVector<float>& widthVec, Qt3DCore::QEntity *_rootEntity, Qt3DRender::QCamera *camera, QString nameWell, bool modeFilaire, bool showNormals)
{
    if(positionsVec.size() < 2)
    {
        qWarning() << "Qt3DHelpers::drawExtrudersNew: insufficient number of input points.";
        return new Qt3DCore::QEntity(_rootEntity);
    }

    int nbpoints = 0;

   // qDebug()<<" position vec :"<<positionsVec.size();
    auto *geometry = new Qt3DCore::QGeometry(_rootEntity);

    int numInputPoints = positionsVec.size();
    int resolution = 1;//30
    int numRotations = 8;//30

  //  QVector<QVector<QVector3D> > offsettPositionsVec;
    QVector<QVector3D> resampledPositionsVec;
    QVector<QVector3D> resampledColorVec;
    QVector<float> resampledWidthVec;


  //  QVector<QVector3D> myPositionsVec;

    for (int i = 0; i < (numInputPoints-1); ++i)
    {
        QVector3D aPos,bPos,cPos,dPos;
        float aWidth,bWidth,cWidth,dWidth;


        bPos = positionsVec[i];
        cPos = positionsVec[i+1];

        bWidth = widthVec[i];
        cWidth = widthVec[i+1];

        if(i > 0) {
            aPos = positionsVec[i-1];
            aWidth = widthVec[i-1];
        } else {
            aPos = positionsVec[i];
            aWidth = widthVec[i];
        }
        if(i < numInputPoints - 2) {
            dPos = positionsVec[i+2];
            dWidth = widthVec[i+2];
        } else {
            dPos = positionsVec[i+1];
            dWidth = widthVec[i+1];
        }

        // linear interpolation
        if(i==0 || bWidth==cWidth)
        {
            for (int j = 0; j < resolution; ++j)
            {
                float value = (float)j / resolution;

                QVector3D resampledPosition = (1.0-value) * bPos + value * cPos;
                float width = (1.0-value) * bWidth + value * cWidth;

                resampledPositionsVec.push_back(resampledPosition);
                resampledWidthVec.push_back(width);
                resampledColorVec.push_back(colorVec[i]);
            }
        }
        // cubic interpolation
        else
        {
            for (int j = 0; j < resolution; ++j)
            {
                float value = (float)j / resolution;

                QVector3D resampledPosition = mix(aPos,bPos,cPos,dPos, value);
                float width = mix(aWidth,bWidth,cWidth,dWidth, value);

                resampledPositionsVec.push_back(resampledPosition);
                resampledWidthVec.push_back(width);
                resampledColorVec.push_back(colorVec[i]);

            }
        }
    }
    resampledPositionsVec.push_back(positionsVec.last());
    resampledWidthVec.push_back(widthVec.last());
    resampledColorVec.push_back(colorVec.last());

    //qDebug() << "Resampling done.";

    int numResampled = resampledPositionsVec.size();
    QVector<QVector3D> resampledTangentVec;
    QVector<QVector3D> resampledNormalVec;

    for (int i = 0; i < numResampled; ++i)
    {
        QVector3D tangent;
        if(i < (numResampled - 1)) {
            tangent = (resampledPositionsVec[i+1] - resampledPositionsVec[i]).normalized();
        } else
        {
            tangent = (resampledPositionsVec[i] - resampledPositionsVec[i-1]).normalized();
        }

        QVector3D normal = computePerpendicularVector(tangent).normalized();

        resampledTangentVec.push_back(tangent);
        resampledNormalVec.push_back(normal);
    }

    // Rotation Minimizing Frames for smooth normals
    // https://www.microsoft.com/en-us/research/wp-content/uploads/2016/12/Computation-of-rotation-minimizing-frames.pdf
    for (int i = 0; i < (numResampled-1); ++i)
    {
        QVector3D t_i = resampledTangentVec[i];
        QVector3D t_i1 = resampledTangentVec[i+1];
        QVector3D r_i = resampledNormalVec[i];

        QVector3D v_1 = resampledPositionsVec[i + 1] - resampledPositionsVec[i];
        float c1 = QVector3D::dotProduct(v_1, v_1);
        QVector3D r_L_i = r_i - (2.0/c1) * QVector3D::dotProduct(v_1, r_i) * v_1;
        QVector3D t_L_i = t_i - (2.0/c1) * QVector3D::dotProduct(v_1, t_i) * v_1;
        QVector3D v_2  = t_i1 - t_L_i;
        float c2 = QVector3D::dotProduct(v_2, v_2);
        QVector3D r_i1 = r_L_i - (2.0/c2) * QVector3D::dotProduct(v_2, r_L_i) * v_2;

        resampledNormalVec[i + 1] = r_i1;


    }

    int numVertices = numResampled * numRotations;
    int numIndices = (numResampled - 1) * numRotations * 6;

   // qDebug()<<" nb vertices :"<<numVertices;
    // position vertices
    QByteArray positionBytes;
    positionBytes.resize(3 * numVertices * sizeof(float));
    float *positions = reinterpret_cast<float*>(positionBytes.data());

    QByteArray normalBytes;
    normalBytes.resize(3 * numVertices * sizeof(float));
    float *normals = reinterpret_cast<float*>(normalBytes.data());

    QByteArray colorBytes;
    colorBytes.resize(3 * numVertices * sizeof(float));
    float *colors = reinterpret_cast<float*>(colorBytes.data());

    // connectivity between vertices
    QByteArray indexBytes;
    indexBytes.resize(numIndices * sizeof(unsigned int));
    unsigned int *indices = reinterpret_cast<unsigned int*>(indexBytes.data());

    QVector<QVector3D> normalsVec;

    for (int i = 0; i < numResampled; ++i)
    {
        QVector3D tangent;

        if(i < (numResampled - 1)) {
            tangent = (resampledPositionsVec[i+1] - resampledPositionsVec[i]).normalized();
        } else
        {
            tangent = (resampledPositionsVec[i] - resampledPositionsVec[i-1]).normalized();
        }

        QVector3D normal = resampledNormalVec[i];
        QVector3D binormal = QVector3D::crossProduct(tangent, normal);

        float rise = 0;
        float run = 0;
        if(i == 0) {
            rise = resampledWidthVec[i+1] - resampledWidthVec[i];
            run = (resampledPositionsVec[i+1] - resampledPositionsVec[i]).length();
        } else {
            if(i < (numResampled-1)) {
                rise = resampledWidthVec[i+1] - resampledWidthVec[i-1];
                run = (resampledPositionsVec[i+1] - resampledPositionsVec[i-1]).length();
            } else {
                rise = resampledWidthVec[i] - resampledWidthVec[i-1];
                run = (resampledPositionsVec[i] - resampledPositionsVec[i-1]).length();
            }
        }

        float alpha = qAtan(rise/run);
        QMatrix4x4 m;
        m.setToIdentity();
        m.rotate(qRadiansToDegrees(alpha), binormal);

        QVector3D surfaceNormal = (m * QVector4D(normal, 1.0)).toVector3D();

        for (int j = 0; j < numRotations; j++)
        {
            float angle = -(360.0f / numRotations) * j;
            QMatrix4x4 m;
            m.setToIdentity();
            m.rotate(angle , tangent);

            QVector3D rotatedNormal =  (m * QVector4D(normal, 1.0)).toVector3D();
            QVector3D rotatedSurfaceNormal =  (m * QVector4D(surfaceNormal, 1.0)).toVector3D();
            rotatedSurfaceNormal = rotatedSurfaceNormal.normalized();

            QVector3D surfacePoint = resampledPositionsVec[i] + rotatedNormal * resampledWidthVec[i];


          /* if(j==0 && resampledWidthVec[i] != 40)
		   {
        	   //qDebug()<<i<<" resampledWidthVec[i] : "<<resampledWidthVec[i];
        	   myPositionsVec.push_back(surfacePoint);
		   }*/

            *positions++ = surfacePoint.x();
            *positions++ = surfacePoint.y();
            *positions++ = surfacePoint.z();

            nbpoints++;
            QVector3D color = resampledColorVec[i];
            *colors++ = color.x();
            *colors++ = color.y();
            *colors++ = color.z();

            if(showNormals)
            {
				QVector3D posDepartNormal = surfacePoint;
				QVector3D posDestinationNormal = surfacePoint+rotatedSurfaceNormal*20.0f;
				normalsVec.push_back(posDepartNormal);
				normalsVec.push_back(posDestinationNormal);
            }

            // ===================

            *normals++ = rotatedSurfaceNormal.x();
            *normals++ = rotatedSurfaceNormal.y();
            *normals++ = rotatedSurfaceNormal.z();
        }
    }

    // draw normal
	  if(showNormals)
	  {


		drawNormals(normalsVec,Qt::green, _rootEntity,1);
	  }

  //  qDebug() <<modeFilaire<< "Extruding done.nbpoints :"<<nbpoints;

    for (int i = 0; i < (numResampled-1); ++i)
    {
        for (int j = 0; j < numRotations; ++j)
        {
            int indexA0 = 0, indexA1 = 0, indexA2 = 0, indexB0 = 0, indexB1 = 0, indexB2 = 0;
            if(j < (numRotations - 1))
            {
                // triangle A
                indexA0 = i * numRotations + j;
                indexA1 = (i+1) * numRotations + j;
                indexA2 = (i+1) * numRotations + j + 1;

                // triangle B
                indexB0 = i * numRotations + j;
                indexB1 = (i+1) * numRotations + j + 1;
                indexB2 = i * numRotations + j + 1;
            }
            else
            {
                // triangle A
                indexA0 = i * numRotations + j;
                indexA1 = (i+1) * numRotations + j;
                indexA2 = (i+1) * numRotations;

                // triangle B
                indexB0 = i * numRotations + j;
                indexB1 = (i+1) * numRotations;
                indexB2 = i * numRotations;
            }

            *indices++ = indexA0;
            *indices++ = indexA1;
            *indices++ = indexA2;

            *indices++ = indexB0;
            *indices++ = indexB1;
            *indices++ = indexB2;
        }
    }
    //qDebug() << "Indexing done.";

    // set index data
    auto *indexBuffer = new Qt3DCore::QBuffer(geometry);
    indexBuffer->setData(indexBytes);

    // set position data
    auto *vertexBuffer = new Qt3DCore::QBuffer(geometry);
    vertexBuffer->setData(positionBytes);

    // set normal data
    auto *normalBuffer = new Qt3DCore::QBuffer(geometry);
    normalBuffer->setData(normalBytes);

    // set color data
    auto *colorBuffer = new Qt3DCore::QBuffer(geometry);
    colorBuffer->setData(colorBytes);

    auto *indexAttribute = new Qt3DCore::QAttribute(geometry);
    indexAttribute->setVertexBaseType(Qt3DCore::QAttribute::UnsignedInt);
    indexAttribute->setAttributeType(Qt3DCore::QAttribute::IndexAttribute);
    indexAttribute->setBuffer(indexBuffer);
    indexAttribute->setCount(numIndices);
    geometry->addAttribute(indexAttribute); // We add the indices linking the points in the geometry

    auto *positionAttribute = new Qt3DCore::QAttribute(geometry);
    positionAttribute->setName(Qt3DCore::QAttribute::defaultPositionAttributeName());
    positionAttribute->setVertexBaseType(Qt3DCore::QAttribute::Float);
    positionAttribute->setVertexSize(3);
    positionAttribute->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
    positionAttribute->setBuffer(vertexBuffer);
    positionAttribute->setByteStride(3 * sizeof(float));
    positionAttribute->setCount(numVertices);
    geometry->addAttribute(positionAttribute); // We add the vertices in the geometry

    auto *normalAttribute = new Qt3DCore::QAttribute(geometry);
    normalAttribute->setName(Qt3DCore::QAttribute::defaultNormalAttributeName());
    normalAttribute->setVertexBaseType(Qt3DCore::QAttribute::Float);
    normalAttribute->setVertexSize(3);
    normalAttribute->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
    normalAttribute->setBuffer(normalBuffer);
    normalAttribute->setByteStride(3 * sizeof(float));
    normalAttribute->setCount(numVertices);
    geometry->addAttribute(normalAttribute);


    auto *colorAttribute = new Qt3DCore::QAttribute(geometry);
    colorAttribute->setName(Qt3DCore::QAttribute::defaultColorAttributeName());
    colorAttribute->setVertexBaseType(Qt3DCore::QAttribute::Float);
    colorAttribute->setVertexSize(3);
    colorAttribute->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
    colorAttribute->setBuffer(colorBuffer);
    colorAttribute->setByteStride(3 * sizeof(float));
    colorAttribute->setCount(numVertices);
    geometry->addAttribute(colorAttribute);


    // mesh
    auto *line = new Qt3DRender::QGeometryRenderer(_rootEntity);
    line->setGeometry(geometry);
    if(modeFilaire)
    	line->setPrimitiveType(Qt3DRender::QGeometryRenderer::Lines);   //Triangles);
    else
    	line->setPrimitiveType(Qt3DRender::QGeometryRenderer::Triangles);
    //line->setPrimitiveType(Qt3DRender::QGeometryRenderer::Points);

    // material
 //   auto *material = new Qt3DRender::QMaterial();
 /*   material->setEffect(
            Qt3DHelpers::generateImageEffect("qrc:/shaders/qt3d/materials/simplePhong.frag",
                    "qrc:/shaders/qt3d/materials/simplePhong.vert"));*/
    // entity
    auto *lineEntity = new Qt3DCore::QEntity(_rootEntity);
    lineEntity->addComponent(line);
 //   lineEntity->addComponent(material);

    lineEntity->setObjectName(nameWell);

    // picker
   /* sPicker = new Qt3DRender::QObjectPicker(lineEntity);
    sPickingSettings = new Qt3DRender::QPickingSettings(sPicker);
    sPickingSettings->setPickMethod(Qt3DRender::QPickingSettings::TrianglePicking);
    sPickingSettings->setPickResultMode(Qt3DRender::QPickingSettings::NearestPick);
    sPickingSettings->setFaceOrientationPickingMode(Qt3DRender::QPickingSettings::FrontAndBackFace);
    sPickingSettings->setEnabled(true);
    sPicker->setEnabled(true);
    sPicker->setHoverEnabled(true);
    lineEntity->addComponent(sPicker);


    // there may appear an issue with dynamically created entities and picking
    // https://stackoverflow.com/questions/60509246/objectpicker-doesnt-work-for-a-dynamically-created-entity-if-there-isnt-alread
    // might need to remove hoverEnabled: true from Scene in qml
    sCamera = camera;


    connect(sPicker, &Qt3DRender::QObjectPicker::pressed, [&](Qt3DRender::QPickEvent* e) {
        qDebug() << "======= Qt3DRender::QObjectPicker::clicked =======";
        // activate on left mouse button
        if(e->button() == Qt3DRender::QPickEvent::Buttons::LeftButton)
        {

        	qDebug() << "picking sur un puits!!!"<<e->entity()->objectName();
            auto p = dynamic_cast<Qt3DRender::QPickTriangleEvent*>(e);
            if(p) {
                QVector3D pos = p->worldIntersection();
                //sCamera->setViewCenter(pos);
                //sCamera->setUpVector(QVector3D(0,-1,0));
            }
            else
            {
                qWarning() << "QPickEvent not of type QPickTriangleEvent.";
            }
        }
    });*/
    return lineEntity;
}

Qt3DCore::QEntity* Qt3DHelpers::loadObj(const char* filename,Qt3DCore::QEntity* _rootEntity)
{
	//qDebug()<<"Qt3DHelpers::loadObj : "<<filename;
	//provisoire
	//int nbvertex=49362;
	//int nbTri=76977;//230931



	// position vertices (start and end)
	/*QByteArray bufferBytes;
	bufferBytes.resize(3 * nbvertex * sizeof(float)); // start.x, start.y, start.end + end.x, end.y, end.z
	float *positions = reinterpret_cast<float*>(bufferBytes.data());

	QByteArray indexBytes;
	indexBytes.resize(3*nbTri * sizeof(unsigned int)); // start to end
	unsigned int *indices = reinterpret_cast<unsigned int*>(indexBytes.data());

	QByteArray uvBytes;
	uvBytes.resize(2*nbvertex * sizeof(float));
	float* uvtex = reinterpret_cast<float*>(uvBytes.data());


	  QByteArray normalBytes;
	normalBytes.resize(3 * nbvertex * sizeof(float));
	float *normals = reinterpret_cast<float*>(normalBytes.data());

	*/

	std::vector<float> normalsV;
	std::vector<float> positionsV;
	std::vector<float> uvV;
	std::vector<unsigned int> indicesV;


	//printf ( "Loading Objects %s ... \n",filename);
	FILE* fn;
	if(filename==NULL)		return nullptr;
	if((char)filename[0]==0)	return nullptr;
	if ((fn = fopen(filename, "rb")) == NULL)
	{
		printf ( "File %s not found!\n" ,filename );
		return nullptr;
	}
	char line[1000];
	memset ( line,0,1000 );
	int vertex_cnt = 0;
/*	int material = -1;
	//std::map<std::string, int> material_map;
	QVector<QVector3D> uvs;
	QVector<QVector<int> > uvMap;

	int indexVert = 0;
	int indexUV=0;
	int indexN =0;
	int indexIndic =0;
	float* rawVertexArray;
	float* rawTexArray;
	float* rawNormalArray;
	uint* rawIndexArray;
*/

	char nameMat[30];

//	qDebug()<<" open obj : "<<filename;
	while(fgets( line, 1000, fn ) != NULL)
	{
		//Vertex v;
		float vpx,vpy,vpz;
		float uvx,uvy,uvz;
		float nx,ny,nz;
		///int sizeV;

		/*if(line[0] == 's' && line[1] == 'i' && line[1] == 'z' )
		{
			if(sscanf(line,"size %d",&sizeV))
			{
				qDebug()<<" sizeV ==>"<<sizeV;
				nbvertex = sizeV;
				nbTri
				bufferBytes.resize(3 * nbvertex * sizeof(float)); // start.x, start.y, start.end + end.x, end.y, end.z
				float *positions = reinterpret_cast<float*>(bufferBytes.data());


				indexBytes.resize(3*nbTri * sizeof(unsigned int)); // start to end
				unsigned int *indices = reinterpret_cast<unsigned int*>(indexBytes.data());


				uvBytes.resize(2*nbvertex * sizeof(float));
				float* uvtex = reinterpret_cast<float*>(uvBytes.data());



				normalBytes.resize(3 * nbvertex * sizeof(float));
				float *normals = reinterpret_cast<float*>(normalBytes.data());

			}
		}*/

		if ( line[0] == 'u' && line[1] == 's' && line[2] == 'e')
		{

			//string s1,s2;
			char s1[25];

			if(sscanf(line,"%s %s %f",&s1,&nameMat)==2)
			{
				qDebug()<<" usemtl ==>"<<nameMat;
			}

		}

		if ( line[0] == 'v' && line[1] == 'n' )
		{
			if(sscanf(line,"vn %f %f %f",&nx,&ny,&nz)==3)
			{
			//	*normals++ = nx;
			//	*normals++ = ny;
			//	*normals++ = nz;

				normalsV.push_back(nx);
				normalsV.push_back(ny);
				normalsV.push_back(nz);


			}
		}
		else if ( line[0] == 'v' && line[1] == 't' )
		{
			if ( line[2] == ' ' )
			if(sscanf(line,"vt %f %f",
				&uvx,&uvy)==2)
			{

			//	*uvtex++ = uvx;
				//*uvtex++ = uvy;

				uvV.push_back(uvx);
				uvV.push_back(uvy);



			} else
			if(sscanf(line,"vt %f %f %f",
				&uvx,&uvy,&uvz)==3)
			{

			//	*uvtex++ = uvx;
			//	*uvtex++ = uvy;
				uvV.push_back(uvx);
				uvV.push_back(uvy);
				//rawTexArray[indexUV++] = uvx;
				//rawTexArray[indexUV++] = uvy;
				//QVector3D uv(uvx,uvy,uvz);
				//uvs.push_back(uv);
			}
		}
		else if ( line[0] == 'v' )
		{
			if ( line[1] == ' ' )
			if(sscanf(line,"v %f %f %f",
				&vpx,	&vpy,	&vpz)==3)
			{
			//	*positions++ = vpx;
			//	*positions++ = vpy;
			//	*positions++ = vpz;

				positionsV.push_back(vpx);
				positionsV.push_back(vpy);
				positionsV.push_back(vpz);



			}
		}
		int integers[9];
		if ( line[0] == 'f' )
		{
			//Triangle t;
			bool tri_ok = false;
			bool has_uv = false;

			if(sscanf(line,"f %d %d %d",
				&integers[0],&integers[1],&integers[2])==3)
			{
				tri_ok = true;
			}else
			if(sscanf(line,"f %d// %d// %d//",
				&integers[0],&integers[1],&integers[2])==3)
			{
				tri_ok = true;
			}else
			if(sscanf(line,"f %d//%d %d//%d %d//%d",
				&integers[0],&integers[3],
				&integers[1],&integers[4],
				&integers[2],&integers[5])==6)
			{
				tri_ok = true;
			}else
			if(sscanf(line,"f %d/%d/%d %d/%d/%d %d/%d/%d",
				&integers[0],&integers[6],&integers[3],
				&integers[1],&integers[7],&integers[4],
				&integers[2],&integers[8],&integers[5])==9)
			{
				tri_ok = true;
				has_uv = true;
			}else // Add Support for v/vt only meshes
			if (sscanf(line, "f %d/%d %d/%d %d/%d",
				&integers[0], &integers[6],
				&integers[1], &integers[7],
				&integers[2], &integers[8]) == 6)
			{
				tri_ok = true;
				has_uv = true;
			}
			else
			{
				printf("unrecognized sequence\n");
				printf("%s\n",line);
				while(1);
			}
			if ( tri_ok )
			{
				//*indices++ =1* (integers[0]-1-vertex_cnt);

				indicesV.push_back((integers[0]-1-vertex_cnt));
			//	*indices++ =1* (integers[1]-1-vertex_cnt);

				indicesV.push_back((integers[1]-1-vertex_cnt));
			//	*indices++ =1* (integers[2]-1-vertex_cnt);
				indicesV.push_back((integers[2]-1-vertex_cnt));


			}
		}
	}


	fclose(fn);

	int nbvertex =positionsV.size()/3;
	int nbTri = indicesV.size()/3;

	QByteArray bufferBytes;
		bufferBytes.resize(3 *nbvertex * sizeof(float)); // start.x, start.y, start.end + end.x, end.y, end.z

		memcpy(bufferBytes.data(),positionsV.data(),3*nbvertex * sizeof(float));
	//	float *positions = reinterpret_cast<float*>(bufferBytes.data());

		QByteArray indexBytes;
		indexBytes.resize(3*nbTri * sizeof(unsigned int)); // start to end
		memcpy(indexBytes.data(),indicesV.data(),3*nbTri * sizeof(unsigned int));
	//	unsigned int *indices = reinterpret_cast<unsigned int*>(indexBytes.data());

		QByteArray uvBytes;
		uvBytes.resize(2*nbvertex * sizeof(float));
		memcpy(uvBytes.data(),uvV.data(),2*nbvertex * sizeof(float));
	//	float* uvtex = reinterpret_cast<float*>(uvBytes.data());


		  QByteArray normalBytes;
		normalBytes.resize(3*nbvertex * sizeof(float));
		memcpy(normalBytes.data(),normalsV.data(),3*nbvertex * sizeof(float));
	//	float *normals = reinterpret_cast<float*>(normalBytes.data());


	auto *geometry = new Qt3DCore::QGeometry(_rootEntity);

	auto *buf = new Qt3DCore::QBuffer(geometry);
	buf->setData(bufferBytes);

	auto *positionAttribute = new Qt3DCore::QAttribute(geometry);
	positionAttribute->setName(Qt3DCore::QAttribute::defaultPositionAttributeName());
	positionAttribute->setVertexBaseType(Qt3DCore::QAttribute::Float);
	positionAttribute->setVertexSize(3);
	positionAttribute->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
	positionAttribute->setBuffer(buf);
	positionAttribute->setByteStride(3 * sizeof(float));
	positionAttribute->setCount(nbvertex);
	geometry->addAttribute(positionAttribute); // We add the vertices in the geometry


	auto *indexBuffer = new Qt3DCore::QBuffer(geometry);
	indexBuffer->setData(indexBytes);

	auto *indexAttribute = new Qt3DCore::QAttribute(geometry);
	indexAttribute->setVertexBaseType(Qt3DCore::QAttribute::UnsignedInt);
	indexAttribute->setAttributeType(Qt3DCore::QAttribute::IndexAttribute);
	indexAttribute->setBuffer(indexBuffer);
	indexAttribute->setCount(nbTri*3);
	geometry->addAttribute(indexAttribute); // We add the indices linking the points in the geometry

	auto *normalBuffer = new Qt3DCore::QBuffer(geometry);
	normalBuffer->setData(normalBytes);

	auto *normalAttribute = new Qt3DCore::QAttribute(geometry);
	normalAttribute->setName(Qt3DCore::QAttribute::defaultNormalAttributeName());
	normalAttribute->setVertexBaseType(Qt3DCore::QAttribute::Float);
	normalAttribute->setVertexSize(3);
	normalAttribute->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
	normalAttribute->setBuffer(normalBuffer);
	normalAttribute->setByteStride(3 * sizeof(float));
	normalAttribute->setCount(nbvertex);
	geometry->addAttribute(normalAttribute);


	Qt3DCore::QAttribute *m_textureAttribute;

    auto *texBuffer = new Qt3DCore::QBuffer(geometry);
    auto *texAttribute = new Qt3DCore::QAttribute(geometry);

    texBuffer->setData(uvBytes);
   // texBuffer->setType(Qt3DCore::QBuffer::VertexBuffer);
    texAttribute->setName(Qt3DCore::QAttribute::defaultTextureCoordinateAttributeName());
    texAttribute->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
  //  texAttribute->setDataType(Qt3DCore::QAttribute::Float);
    texAttribute->setBuffer(texBuffer);
  //  texAttribute->setDataSize(2);
    texAttribute->setByteOffset(0);
    texAttribute->setByteStride(0);
    texAttribute->setCount(nbvertex);

	 geometry->addAttribute(texAttribute);



	auto *objet = new Qt3DRender::QGeometryRenderer(_rootEntity);
	objet->setGeometry(geometry);
	objet->setPrimitiveType(Qt3DRender::QGeometryRenderer::Triangles);   //Triangles);


   auto *objEntity = new Qt3DCore::QEntity(_rootEntity);

   objEntity->addComponent(objet);

   return objEntity;


}



//void Qt3DHelpers::writeObj(const char* filename, QVector<Triangle> triangles,QVector<Vertex> vertices)
void Qt3DHelpers::writeObj(const char* filename, std::vector<QVector3D> vertices,std::vector<QVector3D> normals, std::vector<int> indices)
{
	FILE *file=fopen(filename, "w");
	if (!file)
	{
		printf("write_obj: can't write data file \"%s\".\n", filename);
		exit(0);
	}

	//fprintf(file,size "%d",vertices.size());

	for(int i=0;i<vertices.size();i++)
	{
		fprintf(file, "v %g %g %g\n", vertices[i].x(),vertices[i].y(),vertices[i].z()); //more compact: remove trailing zeros
	}

	for(int i=0;i<vertices.size();i++)
	{
			fprintf(file, "vt %g %g\n", 0.0,0.0);
	}

	for(int i=0;i<normals.size();i++)
	{
		fprintf(file, "vn %g %g %g\n", normals[i].x(),normals[i].y(),normals[i].z());
	}


	for(int i=0;i<indices.size();i+=3)
	{
		fprintf(file, "f %d/%d/%d %d/%d/%d %d/%d/%d\n", indices[i]+1, indices[i]+1, indices[i]+1, indices[i+1]+1, indices[i+1]+1,indices[i+1]+1, indices[i+2]+1, indices[i+2]+1, indices[i+2]+1);
	}
	fclose(file);

}

void Qt3DHelpers::writeObjWidthUV(const char* filename, std::vector<QVector3D> vertices,std::vector<QVector3D> normals,std::vector<QVector2D> uvs, std::vector<int> indices)
{
	FILE *file=fopen(filename, "w");
	if (!file)
	{
		printf("write_obj: can't write data file \"%s\".\n", filename);
		exit(0);
	}


	for(int i=0;i<vertices.size();i++)
	{
		fprintf(file, "v %g %g %g\n", vertices[i].x(),vertices[i].y(),vertices[i].z()); //more compact: remove trailing zeros
	}

	for(int i=0;i<vertices.size();i++)
	{
			fprintf(file, "vt %g %g\n", uvs[i].x(),uvs[i].y());
	}

	for(int i=0;i<normals.size();i++)
	{
		fprintf(file, "vn %g %g %g\n", normals[i].x(),normals[i].y(),normals[i].z());
	}


	for(int i=0;i<indices.size();i+=3)
	{
		fprintf(file, "f %d/%d/%d %d/%d/%d %d/%d/%d\n", indices[i]+1, indices[i]+1, indices[i]+1, indices[i+1]+1, indices[i+1]+1,indices[i+1]+1, indices[i+2]+1, indices[i+2]+1, indices[i+2]+1);
	}
	fclose(file);

}


QVector3D Qt3DHelpers::screenToWorld( QMatrix4x4 viewMatrix, QMatrix4x4 projecMatrix,QVector2D pos2D,int screenWidth,int screenHeight)
{

	QVector3D pos1(0.0f,0.0f,0.0f);
	pos1 = pos1.project(viewMatrix,projecMatrix, QRect(0,0,screenWidth,screenHeight));
	QVector3D pos2(pos2D.x(), screenHeight-pos2D.y(),pos1.z());
	QVector3D worldpos =pos2.unproject(viewMatrix,projecMatrix, QRect(0,0,screenWidth,screenHeight));
	return worldpos;

	/*float xx = 2.0f * pos2D.x() / screenWidth -1;
	float yy = 2.0f * pos2D.y() / screenHeight -1;


	QMatrix4x4 projView = projecMatrix* viewMatrix;
	QMatrix4x4 invProjView = projView.inverted();


	QVector4D pos4D(xx,-yy,-1.0f,1.0f);
	QVector4D worldpos = invProjView * pos4D;
	return QVector3D(worldpos.x(),worldpos.y(),worldpos.z());*/

}

QVector2D Qt3DHelpers::worldToScreen(QVector3D pos, QMatrix4x4 viewMatrix, QMatrix4x4 projecMatrix,int screenWidth,int screenHeight)
{
	QVector3D pos1 = viewMatrix * pos;
	QVector3D pos2 = projecMatrix * pos1;

	pos2.setX(pos2.x() / pos2.z());
	pos2.setY(pos2.y() / pos2.z());

	pos2.setX((pos2.x()+ 1 ) *screenWidth / 2.0f);
	pos2.setY((pos2.y()+ 1 ) * screenHeight / 2.0f);

	pos2.setY(screenHeight - pos2.y());
	return QVector2D(pos2.x(),pos2.y());

	/*pos = Vector3.Transform(pos, viewMatrix);
	pos = Vector3.Transform(pos, projectionMatrix);
	pos.X /= pos.Z;
	pos.Y /= pos.Z;
	pos.X = (pos.X + 1) * screenWidth / 2;
	pos.Y = (pos.Y + 1) * screenHeight / 2;

	return new Vector2(pos.X, pos.Y);*/
}

