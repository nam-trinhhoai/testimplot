#include "layerrgtrep.h"

#include "cudaimagepaletteholder.h"
#include "layerrgtproppanel.h"
#include "layerrgtlayer.h"
#include "layerrgt3Dlayer.h"
#include "qgllineitem.h"
#include "datacontroler.h"
#include "slicepositioncontroler.h"
#include "LayerSlice.h"
#include "seismic3dabstractdataset.h"
#include "seismic3ddataset.h"
#include "affinetransformation.h"
#include "abstractinnerview.h"
#include "cubeseismicaddon.h"

#include <QAction>
#include <QMenu>

LayerRGTRep::LayerRGTRep(LayerSlice *layerslice, AbstractInnerView *parent) :
		AbstractGraphicRep(parent),  IMouseImageDataProvider() {
	m_layerslice = layerslice;
	m_propPanel = nullptr;
	m_layer = nullptr;
	m_layer3D=nullptr;
	m_showCrossHair = false;
	m_name = m_layerslice->name();

	connect(m_layerslice->image(), SIGNAL(dataChanged()), this,
			SLOT(dataChanged()));
        connect(m_layerslice,SIGNAL(deletedMenu()),this,SLOT(deleteLayerRGTRep()));
}

LayerRGTRep::~LayerRGTRep() {
	if (m_layer != nullptr)
		delete m_layer;
	if (m_layer3D != nullptr)
		delete m_layer3D;
	if (m_propPanel != nullptr)
		delete m_propPanel;
}

// MZR 14072021
void LayerRGTRep::buildContextMenu(QMenu *menu){
	QAction *deleteAction = new QAction(tr("Delete Layers"), this);
	menu->addAction(deleteAction);
	connect(deleteAction, SIGNAL(triggered()), this, SLOT(deleteLayerRGTRep()));
}

void LayerRGTRep::deleteLayerRGTRep(){
	m_parent->hideRep(this);
	emit deletedRep(this);

	// Spectrum layer
	if(m_layerslice->getComputationType() == 1){
		m_layerslice->deleteRgt();
	}

	connect(m_layerslice,nullptr,this,nullptr);
	m_layerslice->deleteRep();

	this->deleteLater();
}

LayerSlice* LayerRGTRep::layerSlice() const {
	return m_layerslice;
}

void LayerRGTRep::dataChanged() {
	if (m_propPanel != nullptr)
		m_propPanel->updatePalette();
	if (m_layer != nullptr)
		m_layer->refresh();
	if (m_layer3D != nullptr)
		m_layer3D->refresh();
}

IData* LayerRGTRep::data() const {
	return m_layerslice;
}

QWidget* LayerRGTRep::propertyPanel() {
	if (m_propPanel == nullptr) {
		m_propPanel = new LayerRGTPropPanel(this,
				m_parent->viewType() == ViewType::View3D, m_parent);
		connect(m_propPanel, &QWidget::destroyed, [this]() {
			m_propPanel = nullptr;
		});
	}
	return m_propPanel;
}
GraphicLayer* LayerRGTRep::layer(QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) {
	if (m_layer == nullptr) {
		m_layer = new LayerRGTLayer(this, scene, defaultZDepth, parent);
		m_layer->showCrossHair(m_showCrossHair);
	}
	return m_layer;
}

Graphic3DLayer* LayerRGTRep::layer3D(QWindow *parent, Qt3DCore::QEntity *root,
		Qt3DRender::QCamera *camera) {
	if (m_layer3D == nullptr) {
		m_layer3D = new LayerRGT3DLayer(this, parent, root, camera);
	}
	return m_layer3D;
}

void LayerRGTRep::showCrossHair(bool val) {
	m_showCrossHair = val;
	m_layer->showCrossHair(m_showCrossHair);
}

bool LayerRGTRep::crossHair() const {
	return m_showCrossHair;
}


bool LayerRGTRep::mouseData(double x, double y, MouseInfo &info) {
	double value;
	bool valid = IGeorefImage::value(m_layerslice->image(), x, y, info.i,
			info.j, value);
	info.valuesDesc.push_back("Attribute");
	info.values.push_back(value);
	info.depthValue = true;
	IGeorefImage::value(m_layerslice->isoSurfaceHolder(), x, y, info.i, info.j,
			value);
	double realDepth;
	m_layerslice->seismic()->sampleTransformation()->direct(value, realDepth);
	info.depth = realDepth;
	info.depthUnit = m_layerslice->seismic()->cubeSeismicAddon().getSampleUnit();
	return valid;
}

bool LayerRGTRep::setSampleUnit(SampleUnit sampleUnit) {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return list.contains(sampleUnit);
}

QList<SampleUnit> LayerRGTRep::getAvailableSampleUnits() const {
	CubeSeismicAddon addon = m_layerslice->seismic()->cubeSeismicAddon();
	QList<SampleUnit> list;
	list.push_back(addon.getSampleUnit());
	return list;
}

QString LayerRGTRep::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return (list.contains(sampleUnit)) ? "" : "Dataset unit not compatible";
}

AbstractGraphicRep::TypeRep LayerRGTRep::getTypeGraphicRep() {
    return AbstractGraphicRep::Image;
}
