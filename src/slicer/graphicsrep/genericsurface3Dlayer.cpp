#include "genericsurface3Dlayer.h"
#include "cudaimagetexture.h"
#include "iimagepaletteholder.h"
#include <cmath>
#include <QPropertyAnimation>
#include <QMouseEvent>
#include <QEffect>
#include <QObjectPicker>
#include <Qt3DInput/QMouseHandler>
#include <Qt3DRender/QPickEvent>
#include <Qt3DRender/QPickTriangleEvent>
#include <Qt3DCore/QBuffer>
 #include <Qt3DExtras/QPhongMaterial>
 #include <Qt3DExtras/QCylinderMesh>
#include "interpolation.h"
#include <chrono>




GenericSurface3DLayer::GenericSurface3DLayer(QMatrix4x4 scene,QMatrix4x4 object,QString path)
{
	m_transform=nullptr;
	m_sliceEntity = nullptr;
	m_EntityMesh = nullptr;
	m_palette = nullptr;
	m_actifRayon = false;
	m_nullValue = -9999.0;
	m_nullValueActive = true;

	m_material = nullptr;
	m_opacityParameter = nullptr;
	m_isoSurfaceParameter = nullptr;
	m_nullValueParameter = nullptr;
	m_nullValueActiveParameter = nullptr;
	m_genericMaterial = nullptr;

	m_path = path;

	m_sceneTr=scene;
	m_objectTr=object;

}

float GenericSurface3DLayer::distanceSigned(QVector3D positionTr, bool* ok)
{
	if(positionTr.x() >= 0 && positionTr.x() <=m_palette->width()-1 && positionTr.z() >= 0 && positionTr.z() <=m_palette->height()-1)
	{
		double alt=0.0;
		std::vector<BilinearPoint> points =  bilinearInterpolationPoints(positionTr.x(),positionTr.z(),
				0.0,0.0,m_palette->width()-1,m_palette->height()-1,2,2);

		double altTmp = 0.0;
		double ponderation = 0.0;
		for(int i=0;i<points.size();i++)
		{
			bool res2 = m_palette->valueAt(points[i].i, points[i].j ,altTmp);
			if(res2 && altTmp < m_heightThreshold && (!m_nullValueActive || altTmp!=m_nullValue))
			{

				alt += altTmp* points[i].w;
				ponderation +=points[i].w;
			}
		}


	//	bool res = m_palette->valueAt((int)(positionTr.x()),(int)(positionTr.z()),alt);

		if(points.size() > 0 && !qFuzzyIsNull(ponderation))
		{
			alt = alt/ponderation;
			*ok = true;
			if( alt >= m_heightThreshold || (m_nullValueActive && alt==m_nullValue))
			{
				*ok = false;
				return 0.0f;
			}
			return (alt - positionTr.y());
		}
		else
		{
			*ok = false;
			return 0.0f;
		}
	}
	else
	{
		*ok = false;
		return 0.0f;
	}

}

void GenericSurface3DLayer::setOpacity(float val){
	if (m_opacityParameter!=nullptr) {
		m_opacityParameter->setValue(val);
	}
}

void GenericSurface3DLayer::updateTexture(CudaImageTexture *texture,
		IImagePaletteHolder *img) {
	if (texture == nullptr)
		return;

	texture->setData(img->getDataAsByteArray());
}

void GenericSurface3DLayer::reloadFromCache(SurfaceMeshCache& meshCache)
{
	m_EntityMesh->reloadFromCache(meshCache,m_sceneTr);
}

void GenericSurface3DLayer::update(IImagePaletteHolder *palette, QString path, int simplifySteps, int compression)
{
	m_path = path;
	if (m_EntityMesh)
	{
		if (m_nullValueActive)
		{
			m_EntityMesh->activateNullValue(m_nullValue);
		} else
		{
			m_EntityMesh->deactivateNullValue();
		}
		//auto start = std::chrono::steady_clock::now();
		m_EntityMesh->init_obj(m_width, m_depth, palette, m_heightThreshold,m_cubeOrigin,m_cubeScale,m_path,simplifySteps, compression,m_sceneTr,m_objectTr);


	//	qDebug()<<" draw normals:"<<m_EntityMesh->m_listeNormals.size();

	//	if(m_root!=nullptr)	Qt3DHelpers::drawNormals(m_EntityMesh->m_listeNormals,Qt::green,m_root,2);
		//auto end = std::chrono::steady_clock::now();
		//qDebug() << "test init_obj : " << std::chrono::duration<double, std::milli>(end - start).count();
	}
}


void GenericSurface3DLayer::setVisible(bool b)
{
	m_sliceEntity->setEnabled(b);
}

void GenericSurface3DLayer::Show(Qt3DCore::QEntity *root,QMatrix4x4 transformMesh, int width, int depth,CudaImageTexture * cudaSurfaceTexture,  GenericMaterialInitializer* genericMaterial,
		IImagePaletteHolder *palette,float heightThreshold , float cubescale,float cubeorigin,Qt3DRender::QCamera * camera, float opacite,ImageFormats::QSampleType sampleTypeIso, int simplifySteps,int compression/*,Qt3DRender::QLayer* layer*/ )
{

	m_root= root;
	m_palette = palette;
	m_heightThreshold = heightThreshold;
	m_width= width;
	m_depth = depth;
	m_camera =camera;
	m_genericMaterial = genericMaterial;

	m_sliceEntity = new Qt3DCore::QEntity(root);
	m_EntityMesh = new SurfaceMesh();
	m_EntityMesh->setDimensions(QVector2D(width, depth));

	m_EntityMesh->setTransform(transformMesh);

	//Create a material
	m_material = new Qt3DRender::QMaterial();


	//TODO choose vertex according to type
	if (m_palette->sampleType().isFloat()) {
		genericMaterial->initMaterial(m_material,"qrc:/shaders/qt3d/debugPhong.vert");
	} else if (m_palette->sampleType().isSigned()) {
		genericMaterial->initMaterial(m_material,"qrc:/shaders/qt3d/idebugPhong.vert");
	} else {
		genericMaterial->initMaterial(m_material,"qrc:/shaders/qt3d/udebugPhong.vert");
	}


	m_isoSurfaceParameter = new Qt3DRender::QParameter(QStringLiteral("surfaceMap"),cudaSurfaceTexture);
	m_material->addParameter(m_isoSurfaceParameter);
	m_opacityParameter = new Qt3DRender::QParameter(QStringLiteral("opacity"),opacite);
	m_material->addParameter(m_opacityParameter);

	m_material->addParameter(new Qt3DRender::QParameter(QStringLiteral("heightThreshold"),m_heightThreshold));

	m_nullValueParameter = new Qt3DRender::QParameter(QStringLiteral("nullValue"),m_nullValue);
	m_material->addParameter(m_nullValueParameter);
	m_nullValueActiveParameter = new Qt3DRender::QParameter(QStringLiteral("nullValueActive"),m_nullValueActive ? 1.0 : 0.0);
	m_material->addParameter(m_nullValueActiveParameter);


	m_cubeOrigin = cubeorigin;//tbuf[1*4+1] * sampleTransform->b() + tbuf[3*4+1];
	m_cubeScale = cubescale;//tbuf[1*4+1] * sampleTransform->a();

	if (m_nullValueActive)
	{
		m_EntityMesh->activateNullValue(m_nullValue);
	} else
	{
		m_EntityMesh->deactivateNullValue();
	}
	m_EntityMesh->init_obj(width, depth, palette, heightThreshold,m_cubeOrigin,m_cubeScale,m_path,simplifySteps,compression,m_sceneTr,m_objectTr);

	//qDebug()<<" draw normals:"<<m_EntityMesh->m_listeNormals.size();

	//Qt3DHelpers::drawNormals(m_EntityMesh->m_listeNormals,Qt::green,root,2);

	Qt3DExtras::QPhongMaterial* material = new Qt3DExtras::QPhongMaterial(root);
	material->setAmbient(QColor(0, 0, 0, 0));
	material->setDiffuse(QColor(255,255, 255, 255));
	material->setSpecular(QColor(0, 0, 0, 0));

	Qt3DRender::QMaterial *mymaterial = new Qt3DRender::QMaterial();
	//mymaterial->setEffect(Qt3DHelpers::generateImageEffect("qrc:/shaders/qt3d/myphong.frag","qrc:/shaders/qt3d/myphong.vert"));
	mymaterial->setEffect(Qt3DHelpers::generateImageEffect("qrc:/shaders/qt3d/myphong.frag","qrc:/shaders/qt3d/myphong.vert"));
/*	QVector4D pos(0.0f,0.0f,0.0f,1.0f);
	mymaterial->addParameter(new Qt3DRender::QParameter(QStringLiteral("lightPosition"),pos));

	QVector3D col(0.7f,0.7f,0.7f);
	QVector3D col2(0.15f,0.15f,0.15f);
	mymaterial->addParameter(new Qt3DRender::QParameter(QStringLiteral("lightIntensity"),col));
	mymaterial->addParameter(new Qt3DRender::QParameter(QStringLiteral("kd"),col));
	mymaterial->addParameter(new Qt3DRender::QParameter(QStringLiteral("ka"),col2));
	mymaterial->addParameter(new Qt3DRender::QParameter(QStringLiteral("ks"),col2));*/

	QVector3D col(0.7f,0.7f,0.7f);
	QVector3D pos(0.0f,-2000.0f,0.0f);
	mymaterial->addParameter(new Qt3DRender::QParameter(QStringLiteral("lightPosition"),pos));
	mymaterial->addParameter(new Qt3DRender::QParameter(QStringLiteral("colorObj"),col));

	m_transform = new Qt3DCore::QTransform();
	m_transform->setScale3D(QVector3D(1, 1, 1));

	m_sliceEntity->addComponent(m_EntityMesh);
	m_sliceEntity->addComponent(m_material);
	m_sliceEntity->addComponent(m_transform);
	//m_sliceEntity->addComponent(layer);

	// picker
	spicker = new Qt3DRender::QObjectPicker();
	Qt3DRender::QPickingSettings *pickingSettings = new Qt3DRender::QPickingSettings(spicker);
	pickingSettings->setPickMethod(Qt3DRender::QPickingSettings::TrianglePicking);//TrianglePicking
	pickingSettings->setPickResultMode(Qt3DRender::QPickingSettings::NearestPick);
	pickingSettings->setFaceOrientationPickingMode(Qt3DRender::QPickingSettings::FrontFace);//FrontAndBackFace
	pickingSettings->setEnabled(true);
	spicker->setEnabled(true);
	spicker->setDragEnabled(true);
	m_sliceEntity->addComponent(spicker);//m_ghostEntity

//	m_sliceEntity->setObjectName("surface");

	connect(spicker, &Qt3DRender::QObjectPicker::pressed, [&](Qt3DRender::QPickEvent* e) {
	m_actifRayon= true;
	});
	connect(spicker, &Qt3DRender::QObjectPicker::moved, [&](Qt3DRender::QPickEvent* e) {
	m_actifRayon= false;
	});
	connect(spicker, &Qt3DRender::QObjectPicker::clicked, [&](Qt3DRender::QPickEvent* e) {
	//  qDebug() << "////======= m_ghostEntity::clicked ======="<<m_actifRayon;

	if(m_actifRayon== true )//&& e->button() == Qt3DRender::QPickEvent::Buttons::LeftButton)
	{
		QVector3D pos = e->worldIntersection();
		//QVector3D poslocal = e->localIntersection();
	//	QVector3D dir1 = m_camera->viewVector().normalized();
	//	QVector3D dir2 = (pos -m_camera->position()).normalized();

	//	float angle = 180.0f / 3.14159 * acos(QVector3D::dotProduct(dir1,dir2));

		//qDebug()<<"  angle : "<<angle;

		//if(angle <40.0f)
		//{
			float coefZoom = 0.35f;
			QVector3D dirDest = (pos - m_camera->position()) * coefZoom;
			QVector3D  newpos = m_camera->position() + dirDest;


			emit sendPositionCam(e->button(),pos);
			emit sendPositionTarget(newpos,pos);
	/*	}
		else
		{
			qDebug()<<" point incorrect no move camera "<<angle;
		}*/



	}
	});
}


template<typename InputType>
struct UpdateIsoSurfaceKernelGeneric {
	static void run(int meshWidth, int meshHeight, int textureWidth, int textureHeight,
			CUDAImagePaletteHolder* palette, float heightThreshold, float cubeOrigin,
			float cubeScale, float* rawVertexArray){//, float* rawVertexArrayOld) {



		palette->lockPointer();
		InputType* tab = static_cast<InputType*>(palette->backingPointer());

		#pragma omp parallel for
		for (int i = 0; i < meshWidth; i++)
		{
			for (int j = 0; j < meshHeight; j++)
			{
				int bufferIndexMesh = j * meshWidth + i;

				int textureIndexWidth = ((float) i / (float ) meshWidth) * (float) textureWidth;
				textureIndexWidth = fmin(textureIndexWidth, textureWidth - 1);
				int textureIndexHeight = ((float) j / (float ) meshHeight) * (float) textureHeight;
				textureIndexHeight = fmin(textureIndexHeight, textureHeight - 1);



				double heightValue = tab[textureIndexWidth + textureIndexHeight * textureWidth];
				//palette->valueAt(textureIndexWidth, textureIndexHeight, heightValue);
				// workaround for holes: search for valid height in the neighborhood
				int xIndex = textureIndexWidth;
				int yIndex = textureIndexHeight;

				bool isValid=  true;



				if(heightValue > heightThreshold )
				{
					double lastvalue = heightValue;
					int xDir, yDir;
					textureIndexWidth > (textureWidth / 2) ? xDir = -1 : xDir = 1;
					textureIndexHeight > (textureHeight / 2) ? yDir = -1 : yDir = 1;
					xIndex += xDir;
					yIndex += yDir;
					if (xIndex > textureWidth && yIndex > textureHeight) {
						//qDebug()  << "ERROR";
						isValid = false;
//                        errorCountSearchInit++;
					}
					while (xIndex < textureWidth && xIndex>=0 && yIndex < textureHeight && yIndex>=0) {

						//palette->valueAt(xIndex, yIndex, heightValue);
						heightValue = tab[xIndex + yIndex * textureWidth];

						//qDebug() << xIndex << yIndex << heightValue;
						if(heightValue < heightThreshold) {
						//	qDebug()<<" heightThreshold: " <<heightThreshold<< " , last heightValue: " <<lastvalue<< " , new heightValue: " <<heightValue;
							//qDebug() << "success";
							break;
						}
						xIndex += xDir;
						yIndex += yDir;
					}
					if (xIndex >= textureWidth || yIndex >= textureHeight || xIndex<0 || yIndex<0) {
						//qDebug()  << "ERROR";
						isValid = false;
//                        errorCountNoValidHeight++;
					}
				}

				float nbelem= 1.0f;
				if(textureIndexWidth < textureWidth-1)
				{
					if(tab[textureIndexWidth+1 + textureIndexHeight * textureWidth] < heightThreshold)
					{
						heightValue += tab[textureIndexWidth+1 + textureIndexHeight * textureWidth];
						nbelem+=1.0f;
					}
				}
				if(textureIndexWidth > 0)
				{
					if(tab[textureIndexWidth-1 + textureIndexHeight * textureWidth] < heightThreshold)
					{
						heightValue += tab[textureIndexWidth-1 + textureIndexHeight * textureWidth];
						nbelem+=1.0f;
					}
				}
				if(textureIndexHeight > 0)
				{
					if(tab[textureIndexWidth + (textureIndexHeight-1) * textureWidth] < heightThreshold)
					{
						heightValue += tab[textureIndexWidth + (textureIndexHeight-1) * textureWidth];
						nbelem+=1.0f;
					}
				}
				if(textureIndexHeight < textureHeight-1)
				{
					if(tab[textureIndexWidth + (textureIndexHeight+1) * textureWidth] < heightThreshold)
					{
						heightValue += tab[textureIndexWidth + (textureIndexHeight+1) * textureWidth];
						nbelem+=1.0f;
					}
				}
				if(textureIndexWidth < textureWidth-1 && textureIndexHeight > 0)
				{
					if(tab[textureIndexWidth+1 + (textureIndexHeight-1) * textureWidth] < heightThreshold)
					{
						heightValue += tab[textureIndexWidth+1 + (textureIndexHeight-1) * textureWidth];
						nbelem+=1.0f;
					}
				}
				if(textureIndexWidth > 0 && textureIndexHeight < textureHeight-1)
				{
					if(tab[textureIndexWidth-1 + (textureIndexHeight+1) * textureWidth] < heightThreshold)
					{
						heightValue += tab[textureIndexWidth-1 + (textureIndexHeight+1) * textureWidth];
						nbelem+=1.0f;
					}
				}
				if(textureIndexWidth > 0 && textureIndexHeight > 0)
				{
					if(tab[textureIndexWidth-1 + (textureIndexHeight-1) * textureWidth] < heightThreshold)
					{
						heightValue += tab[textureIndexWidth-1 + (textureIndexHeight-1) * textureWidth];
						nbelem+=1.0f;
					}
				}
				if(textureIndexWidth < textureWidth-1 && textureIndexHeight < textureHeight-1)
				{
					if(tab[textureIndexWidth+1 + (textureIndexHeight+1) * textureWidth] < heightThreshold)
					{
						heightValue += tab[textureIndexWidth+1 + (textureIndexHeight+1) * textureWidth];
						nbelem+=1.0f;
					}
				}
				heightValue/= nbelem;
				float newHeightValue =  cubeOrigin + cubeScale *heightValue;
			/*	if (m_rep->fixedRGBLayersFromDataset()->isIsoInT())
					newHeightValue = heightValue;
			 */



			//	rawVertexArray[(bufferIndexMesh * 3)] = rawVertexArrayOld[bufferIndexMesh * 3];
				rawVertexArray[(bufferIndexMesh * 3) + 1] = newHeightValue;
			//   rawVertexArray[(bufferIndexMesh * 3) + 2] = rawVertexArrayOld[bufferIndexMesh * 3 + 2];

			}
		}
		palette->unlockPointer();
	}
};

void GenericSurface3DLayer::zScale(float val)
{
	if (m_transform!=nullptr) {
		m_transform->setScale3D(QVector3D(1, val, 1));
	}
  //  m_ghostTransform->setScale3D(QVector3D(1, val, 1));
}

void GenericSurface3DLayer::hide()
{

	if(m_sliceEntity!= nullptr)
	{
		m_sliceEntity->setParent((Qt3DCore::QEntity*) nullptr);
		m_sliceEntity->deleteLater();
		m_sliceEntity = nullptr;

		m_EntityMesh = nullptr;
		m_transform = nullptr;
		m_material = nullptr;
		m_camera = nullptr;

		m_palette = nullptr; // dereference CUDAImagePaletteHolder

		m_genericMaterial->hide();

		m_opacityParameter = nullptr;
		m_nullValueParameter = nullptr;
		m_nullValueActiveParameter = nullptr;
	}
}

void GenericSurface3DLayer::activateNullValue(float nullValue)
{
	m_nullValueActive = true;
	m_nullValue = nullValue;

	if (m_nullValueParameter) {
		m_nullValueParameter->setValue(m_nullValue);
	}
	if (m_nullValueActiveParameter) {
		m_nullValueActiveParameter->setValue(m_nullValueActive);
	}
}

void GenericSurface3DLayer::deactivateNullValue()
{
	m_nullValueActive = false;

	if (m_nullValueActiveParameter) {
		m_nullValueActiveParameter->setValue(m_nullValueActive);
	}
}

float GenericSurface3DLayer::nullValue() const
{
	return m_nullValue;
}

bool GenericSurface3DLayer::isNullValueActive() const
{
	return m_nullValueActive;
}



