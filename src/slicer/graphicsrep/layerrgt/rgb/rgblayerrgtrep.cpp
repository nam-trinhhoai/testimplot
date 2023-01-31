#include "rgblayerrgtrep.h"
#include "cudaimagepaletteholder.h"
#include "cudargbimage.h"
#include "rgblayerrgtproppanel.h"
#include "rgblayerrgtlayer.h"
#include "rgblayerrgt3dlayer.h"
#include "rgblayerslice.h"
#include "LayerSlice.h"
#include "seismic3dabstractdataset.h"
#include "affinetransformation.h"
#include "abstractinnerview.h"
#include "seismic3ddataset.h"
#include "workingsetmanager.h"
#include "rgbqglcudaimageitem.h"
#include "igeorefimage.h"
#include "affine2dtransformation.h"
#include "cudaimagepaletteholder.h"
#include "basemapsurface.h"

#include <QMenu>

RGBLayerRGTRep::RGBLayerRGTRep(RGBLayerSlice *layerSlice,
		AbstractInnerView *parent) :
		AbstractGraphicRep(parent),  IMouseImageDataProvider() {
	m_rgbLayerSlice = layerSlice;
	m_propPanel = nullptr;
	m_layer = nullptr;
	m_layer3D=nullptr;
	m_name = m_rgbLayerSlice->name();

	connect(m_rgbLayerSlice->image()->get(0), SIGNAL(dataChanged()), this,
			SLOT(dataChangedRed()));
	connect(m_rgbLayerSlice->image()->get(1), SIGNAL(dataChanged()), this,
			SLOT(dataChangedGreen()));
	connect(m_rgbLayerSlice->image()->get(2), SIGNAL(dataChanged()), this,
			SLOT(dataChangedBlue()));

	connect(m_rgbLayerSlice,SIGNAL(deletedRep()),this,SLOT(deleteRGBLayerRGTRep()));// MZR 17082021
}


RGBLayerRGTRep::~RGBLayerRGTRep() {
	if (m_layer != nullptr)
		delete m_layer;
	if (m_layer3D != nullptr)
		delete m_layer3D;
	if (m_propPanel != nullptr)
		delete m_propPanel;
}


RGBLayerSlice* RGBLayerRGTRep::rgbLayerSlice() const {
	return m_rgbLayerSlice;
}

void RGBLayerRGTRep::dataChangedRed() {
	if (m_propPanel != nullptr) {
		m_propPanel->updatePalette(0);
	}
	if (m_layer != nullptr)
		m_layer->refresh();
}

void RGBLayerRGTRep::dataChangedGreen() {
	if (m_propPanel != nullptr) {
		m_propPanel->updatePalette(1);
	}
	if (m_layer != nullptr)
		m_layer->refresh();
}

void RGBLayerRGTRep::dataChangedBlue() {
	if (m_propPanel != nullptr) {
		m_propPanel->updatePalette(2);
	}
	if (m_layer != nullptr)
		m_layer->refresh();
}

IData* RGBLayerRGTRep::data() const {
	return m_rgbLayerSlice;
}

QWidget* RGBLayerRGTRep::propertyPanel() {
	if (m_propPanel == nullptr)
	{
		m_propPanel = new RGBLayerRGTPropPanel(this, m_parent);
		connect(m_propPanel, &QWidget::destroyed, [this]() {
			m_propPanel = nullptr;
		});
	}

	return m_propPanel;
}
GraphicLayer* RGBLayerRGTRep::layer(QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) {
	if (m_layer == nullptr)
	{
		m_layer = new RGBLayerRGTLayer(this, scene, defaultZDepth, parent);
	}

	return m_layer;
}

Graphic3DLayer * RGBLayerRGTRep::layer3D(QWindow * parent,Qt3DCore::QEntity *root,Qt3DRender::QCamera * camera)
{
	if (m_layer3D == nullptr) {
		m_layer3D = new RGBLayerRGT3DLayer(this, parent, root, camera);
	}
	return m_layer3D;
}


bool RGBLayerRGTRep::mouseData(double x, double y, MouseInfo &info) {
	double v1, v2, v3;
	bool valid = IGeorefImage::value(m_rgbLayerSlice->image()->get(0), x, y,
			info.i, info.j, v1);
	valid = IGeorefImage::value(m_rgbLayerSlice->image()->get(1), x, y, info.i,
			info.j, v2);
	valid = IGeorefImage::value(m_rgbLayerSlice->image()->get(2), x, y, info.i,
			info.j, v3);

	info.values.push_back(v1);
	info.values.push_back(v2);
	info.values.push_back(v3);

	info.valuesDesc.push_back("Red");
	info.valuesDesc.push_back("Green");
	info.valuesDesc.push_back("Blue");
	info.depthValue = true;

	IGeorefImage::value(m_rgbLayerSlice->layerSlice()->isoSurfaceHolder(), x, y, info.i, info.j,
			v1);
	double realDepth;
	m_rgbLayerSlice->layerSlice()->seismic()->sampleTransformation()->direct(v1, realDepth);
	info.depth = realDepth;
	info.depthUnit = m_rgbLayerSlice->layerSlice()->seismic()->cubeSeismicAddon().getSampleUnit();

	return valid;
}

void RGBLayerRGTRep::deleteGraphicItemDataContent(QGraphicsItem *item)
{
	deleteData(m_rgbLayerSlice->image()->get(0),item);
	deleteData(m_rgbLayerSlice->image()->get(1),item);
	deleteData(m_rgbLayerSlice->image()->get(2),item);
	deleteData(m_rgbLayerSlice->layerSlice()->isoSurfaceHolder(),item, m_rgbLayerSlice->layerSlice()->height()-1);
}

bool RGBLayerRGTRep::setSampleUnit(SampleUnit sampleUnit) {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return list.contains(sampleUnit);
}

QList<SampleUnit> RGBLayerRGTRep::getAvailableSampleUnits() const {
	CubeSeismicAddon addon = m_rgbLayerSlice->layerSlice()->seismic()->cubeSeismicAddon();
	QList<SampleUnit> list;
	list.push_back(addon.getSampleUnit());
	return list;
}

// MZR 17082021
void RGBLayerRGTRep::buildContextMenu(QMenu * menu){
	QAction *deleteAction = new QAction(tr("Delete Layers"), this);
	menu->addAction(deleteAction);
	connect(deleteAction, SIGNAL(triggered()), this, SLOT(deleteRGBLayerRGTRep()));
}
// MZR 17082021
void RGBLayerRGTRep::deleteRGBLayerRGTRep(){
	m_parent->hideRep(this);
	emit deletedRep(this);

	connect(m_rgbLayerSlice,nullptr,this,nullptr);// MZR 17082021

	WorkingSetManager *manager = m_rgbLayerSlice->workingSetManager();
	manager->deleteRGBLayerSlice(m_rgbLayerSlice);

	this->deleteLater();
}

QString RGBLayerRGTRep::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return (list.contains(sampleUnit)) ? "" : "Dataset unit not compatible";
}

AbstractGraphicRep::TypeRep RGBLayerRGTRep::getTypeGraphicRep() {
    return AbstractGraphicRep::Image;
}

QGraphicsObject* RGBLayerRGTRep::cloneCUDAImageWithMask(QGraphicsItem *parent)
{
	LayerSlice *layer = dynamic_cast<LayerSlice *> (rgbLayerSlice()->layerSlice());
	const float* tabR = nullptr, *tabG = nullptr, *tabB = nullptr, *tabIso = nullptr;
	CUDARGBImage* img = new CUDARGBImage(layer->width(), layer->depth(),
				ImageFormats::QSampleType::FLOAT32, layer->seismic()->ijToXYTransfo());

	// maybe surfaceholder should not be there and replaced by nullptr
	// and allow nullptr isoSurface in RGBQGLCUDAImageItem
	CUDAImagePaletteHolder *surfaceholder = new CUDAImagePaletteHolder(layer->width(), layer->depth(),
			ImageFormats::QSampleType::FLOAT32, layer->seismic()->ijToXYTransfo());

	tabR = layer->getModuleData(rgbLayerSlice()->redIndex());
	tabG = layer->getModuleData(rgbLayerSlice()->greenIndex());
	tabB = layer->getModuleData(rgbLayerSlice()->blueIndex());

	img->get(0)->updateTexture(tabR, false);
	img->get(1)->updateTexture(tabG, false);
	img->get(2)->updateTexture(tabB, false);

	img->get(0)->setRange(rgbLayerSlice()->image()->get(0)->range());
	img->get(1)->setRange(rgbLayerSlice()->image()->get(1)->range());
	img->get(2)->setRange(rgbLayerSlice()->image()->get(2)->range());

	tabIso = layer->getModuleData(0);
	surfaceholder->updateTexture(tabIso, false);
	surfaceholder->setRange(rgbLayerSlice()->layerSlice()->isoSurfaceHolder()->range());


	RGBQGLCUDAImageItem* outItem = new RGBQGLCUDAImageItem(surfaceholder,
				img, 0, parent, true);

	outItem->setMinimumValue(m_rgbLayerSlice->minimumValue());
	outItem->setMinimumValueActive(m_rgbLayerSlice->isMinimumValueActive());

	img->setParent(outItem);
	surfaceholder->setParent(outItem);
	return outItem;
}

BaseMapSurface* RGBLayerRGTRep::cloneCUDAImageWithMaskOnBaseMap(QGraphicsItem *parent) {
	QGraphicsObject* item = cloneCUDAImageWithMask(parent);

	LayerSlice* layer = m_rgbLayerSlice->layerSlice();
	const AffineTransformation* sampleTransfo = layer->getDatasetS()->sampleTransformation();
	CUDAImagePaletteHolder *surfaceholder = new CUDAImagePaletteHolder(layer->width(), layer->depth(),
				ImageFormats::QSampleType::FLOAT32, layer->seismic()->ijToXYTransfo());

	// tabIso is in index not SampleUnit, it need to be converted
	const float* tabIso = layer->getModuleData(0);
	std::vector<float> convertedIso;
	convertedIso.resize(layer->width()* layer->depth());

	long N = convertedIso.size();
	for (long i=0; i<N; i++) {
		double val;
		sampleTransfo->direct(tabIso[i], val);
		convertedIso[i] = val;
	}

	QVector2D oriRange = layer->isoSurfaceHolder()->range();
	double convertedX, convertedY;
	sampleTransfo->direct(oriRange.x(), convertedX);
	sampleTransfo->direct(oriRange.y(), convertedY);
	QVector2D convertedRange = QVector2D(convertedX, convertedY);

	surfaceholder->updateTexture(convertedIso.data(), false);
	surfaceholder->setRange(convertedRange);

	BaseMapSurface* output = new BaseMapSurface(item, surfaceholder, layer->getDatasetS()->cubeSeismicAddon().getSampleUnit());

	return output;
}
