#include "genericsurfacegray3Dlayer.h"
#include "cudaimagetexture.h"
#include "colortabletexture.h"
#include "surfacemeshcacheutils.h"
#include <cmath>
#include <QPropertyAnimation>
#include <QMouseEvent>
#include <QEffect>
#include <QObjectPicker>
#include <Qt3DInput/QMouseHandler>
#include <Qt3DRender/QPickEvent>
#include <Qt3DRender/QPickTriangleEvent>
#include <Qt3DCore/QBuffer>


GenericSurfaceGray3DLayer::GenericSurfaceGray3DLayer(QMatrix4x4 scene,QMatrix4x4 object,QString path)
{
	m_transform=nullptr;
	m_sliceEntity = nullptr;
	m_EntityMesh = nullptr;
	m_palette = nullptr;
	m_actifRayon = false;

	m_material = nullptr;
	m_opacityParameter = nullptr;

	m_paletteAttrRangeParameter = nullptr;

	m_path = path;

		m_sceneTr=scene;
		m_objectTr=object;

}

GenericSurfaceGray3DLayer::~GenericSurfaceGray3DLayer() {
}

void GenericSurfaceGray3DLayer::setOpacity(float val){
	if (m_opacityParameter!=nullptr) {
		m_opacityParameter->setValue(val);
	}

}

void GenericSurfaceGray3DLayer::rangeChanged(const QVector2D &value) {

	if (m_paletteAttrRangeParameter!=nullptr) {
		m_paletteAttrRangeParameter->setValue(value);
	}

}

void GenericSurfaceGray3DLayer::updateTexture(CudaImageTexture *texture,
		CUDAImagePaletteHolder *img) {
	if (texture == nullptr)
		return;

	size_t pointerSize = img->internalPointerSize();
	img->lockPointer();
	texture->setData(
			byteArrayFromRawData((const char*) img->backingPointer(),
					pointerSize));
	img->unlockPointer();
}

void GenericSurfaceGray3DLayer::reloadFromCache(SurfaceMeshCache& meshCache)
{
	m_EntityMesh->reloadFromCache(meshCache,m_sceneTr);
}

void GenericSurfaceGray3DLayer::update(CUDAImagePaletteHolder *palette, QString path, int simplifySteps, int compression)
{
	m_path = path;
	//qDebug()<<" new path :"<<path;

	 m_EntityMesh->init_obj(m_width, m_depth, palette, m_heightThreshold,m_cubeOrigin,m_cubeScale,m_path,simplifySteps, compression,m_sceneTr,m_objectTr);
}

void GenericSurfaceGray3DLayer::Show(Qt3DCore::QEntity *root,QMatrix4x4 transformMesh, int width, int depth,CudaImageTexture * cudaAttrTexture,
		CudaImageTexture * cudaSurfaceTexture, QVector2D ratioAttr, ColorTableTexture * colorTexture, CUDAImagePaletteHolder *palette,
		float heightThreshold , float cubescale,float cubeorigin,Qt3DRender::QCamera * camera, float opacite,
		ImageFormats::QSampleType sampleTypeImage,ImageFormats::QSampleType sampleTypeIso, int simplifySteps,int compression )
{

	m_palette = palette;
	m_heightThreshold = heightThreshold;
	m_camera =camera;
	m_width= width;
	m_depth = depth;

	m_sliceEntity = new Qt3DCore::QEntity(root);
	m_EntityMesh = new SurfaceMesh();
	m_EntityMesh->setDimensions(QVector2D(width, depth));

	m_EntityMesh->setTransform(transformMesh);

	//Create a material
	m_material = new Qt3DRender::QMaterial();

	QString vertexStr;
	if (sampleTypeIso.isFloat()) {
		vertexStr = "qrc:/shaders/qt3d/debugPhong.vert";
	} else if (sampleTypeIso.isSigned()) {
		vertexStr = "qrc:/shaders/qt3d/idebugPhong.vert";
	} else {
		vertexStr = "qrc:/shaders/qt3d/udebugPhong.vert";
	}

	// Set the effect on the materials
	if (sampleTypeImage==ImageFormats::QSampleType::FLOAT32) {
		m_material->setEffect(
				Qt3DHelpers::generateImageEffect("qrc:/shaders/qt3d/grayDebugPhong.frag",
						vertexStr));
	} else {
		m_material->setEffect(
				Qt3DHelpers::generateImageEffect("qrc:/shaders/qt3d/igrayDebugPhong.frag",
						vertexStr));
	}

	m_material->addParameter(
			new Qt3DRender::QParameter(QStringLiteral("elementMap"),cudaAttrTexture));
	m_material->addParameter(
			new Qt3DRender::QParameter(QStringLiteral("surfaceMap"),
					cudaSurfaceTexture));
	m_material->addParameter(
			new Qt3DRender::QParameter(QStringLiteral("colormap"),
					colorTexture));

	m_paletteAttrRangeParameter = new Qt3DRender::QParameter(
				QStringLiteral("paletteRange"),ratioAttr);
	m_material->addParameter(m_paletteAttrRangeParameter);

	m_opacityParameter = new Qt3DRender::QParameter(QStringLiteral("opacity"),opacite);
	m_material->addParameter(m_opacityParameter);

	m_material->addParameter(
			new Qt3DRender::QParameter(QStringLiteral("heightThreshold"),m_heightThreshold));

	m_cubeOrigin = cubeorigin;
	m_cubeScale = cubescale;

//	updateIsoSurface();



	m_EntityMesh->init_obj(width, depth, palette, heightThreshold,m_cubeOrigin,m_cubeScale,m_path,simplifySteps,compression,m_sceneTr,m_objectTr);



	m_transform = new Qt3DCore::QTransform();
	m_transform->setScale3D(QVector3D(1, 1, 1));

	m_sliceEntity->addComponent(m_EntityMesh);
	m_sliceEntity->addComponent(m_material);
	m_sliceEntity->addComponent(m_transform);

	// picker
	Qt3DRender::QObjectPicker *picker = new Qt3DRender::QObjectPicker();
	Qt3DRender::QPickingSettings *pickingSettings = new Qt3DRender::QPickingSettings(picker);
	pickingSettings->setPickMethod(Qt3DRender::QPickingSettings::TrianglePicking);
	pickingSettings->setPickResultMode(Qt3DRender::QPickingSettings::NearestPick);
	pickingSettings->setFaceOrientationPickingMode(Qt3DRender::QPickingSettings::FrontAndBackFace);
	pickingSettings->setEnabled(true);
	picker->setEnabled(true);
	picker->setDragEnabled(true);
	m_sliceEntity->addComponent(picker);//m_ghostEntity

	connect(picker, &Qt3DRender::QObjectPicker::pressed, [&](Qt3DRender::QPickEvent* e) {
		m_actifRayon= true;
	});
	connect(picker, &Qt3DRender::QObjectPicker::moved, [&](Qt3DRender::QPickEvent* e) {
		m_actifRayon= false;
	});
	connect(picker, &Qt3DRender::QObjectPicker::clicked, [&](Qt3DRender::QPickEvent* e) {
		if(m_actifRayon== true && e->button() == Qt3DRender::QPickEvent::Buttons::LeftButton)
		{
			//auto p = dynamic_cast<Qt3DRender::QPickTriangleEvent*>(e);
			//if(p) {
				QVector3D pos = e->worldIntersection();

			/*	float coefZoom = 0.35f;
							QVector3D dirDest = (pos - m_camera->position()) * coefZoom;
							QVector3D  newpos = m_camera->position() + dirDest;


						emit sendPositionCam(e->button(),pos);
						emit sendPositionTarget(newpos,pos);*/

				QPropertyAnimation* animation = new QPropertyAnimation(m_camera,"viewCenter");
				animation->setDuration(2000);
				animation->setStartValue(m_camera->viewCenter());
				animation->setEndValue(pos);
				animation->start();

				float coefZoom = 0.7f;
				QVector3D dirDest = (pos - m_camera->position()) * coefZoom;
				QVector3D  newpos = m_camera->position() + dirDest;

				QPropertyAnimation* animation2 = new QPropertyAnimation(m_camera,"position");
				animation2->setDuration(2000);
				animation2->setStartValue(m_camera->position());
				animation2->setEndValue(newpos);
				animation2->start();
			/*} else {
				qWarning() << "QPickEvent not of type QPickTriangleEvent.";
			}*/
		}
	});

	//connect(m_palette, &CUDAImagePaletteHolder::dataChanged, this, &GenericSurfaceGray3DLayer::updateIsoSurface);
}


template<typename InputType>
struct UpdateIsoSurfaceKernelGenericGray {
	static void run(int meshWidth, int meshHeight, int textureWidth, int textureHeight,
			CUDAImagePaletteHolder* palette, float heightThreshold, float cubeOrigin,
			float cubeScale, float* rawVertexArray){//, float* rawVertexArrayOld) {


		qDebug()<<"============> RUN generic ";

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

				rawVertexArray[(bufferIndexMesh * 3) + 1] = newHeightValue;
			}
		}
		palette->unlockPointer();
	}
};

void GenericSurfaceGray3DLayer::zScale(float val)
{
	if (m_transform!=nullptr) {
		m_transform->setScale3D(QVector3D(1, val, 1));
	}

}

void GenericSurfaceGray3DLayer::hide()
{
	disconnect(m_palette, &CUDAImagePaletteHolder::dataChanged, this, &GenericSurfaceGray3DLayer::updateIsoSurface);

	m_sliceEntity->setParent((Qt3DCore::QEntity*) nullptr);
	m_sliceEntity->deleteLater();
	m_sliceEntity = nullptr;

	m_camera = nullptr;
	m_EntityMesh = nullptr;
	m_transform = nullptr;
	m_material = nullptr;
	m_palette = nullptr; // dereference CUDAImagePaletteHolder
	m_opacityParameter = nullptr;
	m_paletteAttrRangeParameter = nullptr;

}

void GenericSurfaceGray3DLayer::updateIsoSurface()
{
    if (m_palette==nullptr) {
    	return;
    }




	/*int textureWidth = m_palette->width();
    int textureHeight = m_palette->height();

    Qt3DRender::QBuffer* vertexBuffer = m_EntityMesh->getVertexBuffer();
    QByteArray vertexBufferData = vertexBuffer->data();
    auto *rawVertexArray= reinterpret_cast<float*>(vertexBufferData.data());

    int meshWidth = m_EntityMesh->getWidth();
    int meshHeight = m_EntityMesh->getHeight();

    UpdateIsoSurfaceKernelGenericGray<short>::run(meshWidth, meshHeight, textureWidth, textureHeight,
    		m_palette, m_heightThreshold, m_cubeOrigin, m_cubeScale, rawVertexArray);

    vertexBuffer->setData(vertexBufferData);

    m_EntityMesh->computeNormals();*/
}



