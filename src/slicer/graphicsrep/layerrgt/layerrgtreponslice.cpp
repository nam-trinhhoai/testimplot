#include "layerrgtreponslice.h"

#include "cudaimagepaletteholder.h"
#include "layerrgtslicelayer.h"
#include "qgllineitem.h"
#include "LayerSlice.h"
#include "seismic3dabstractdataset.h"
#include "affinetransformation.h"
#include "abstractinnerview.h"
#include "layerrgtproppanelonslice.h"
#include "workingsetmanager.h"
#include <QMenu>
#include <QAction>

LayerRGTRepOnSlice::LayerRGTRepOnSlice(LayerSlice *layerslice, const IGeorefImage * const transfoProvider, SliceDirection dir,AbstractInnerView *parent) :
		AbstractGraphicRep(parent),ISliceableRep(),m_transfoProvider(transfoProvider) {
	m_layerslice = layerslice;
	m_propPanel = nullptr;
	m_layer = nullptr;
	m_dir=dir;
	m_name = m_layerslice->name();
	m_currentSlice=0;

	connect(m_layerslice,SIGNAL(deletedMenu()),this,SLOT(deleteLayerRGTRepOnSlice()));

}

LayerRGTRepOnSlice::~LayerRGTRepOnSlice() {
	if (m_layer != nullptr)
		delete m_layer;
	if (m_propPanel != nullptr)
		delete m_propPanel;
}

LayerSlice* LayerRGTRepOnSlice::layerSlice() const {
	return m_layerslice;
}

void LayerRGTRepOnSlice::setSliceIJPosition(int imageVal)
{
	m_currentSlice=imageVal;
	if(m_layer!=nullptr)
		m_layer->setSliceIJPosition(imageVal);
}

IData* LayerRGTRepOnSlice::data() const {
	return m_layerslice;
}

QWidget* LayerRGTRepOnSlice::propertyPanel() {
	if (m_propPanel == nullptr) {
		m_propPanel = new LayerRGTPropPanelOnSlice(this, m_parent);
		connect(m_propPanel, &QWidget::destroyed, [this]() {
			m_propPanel = nullptr;
		});
	}
	return m_propPanel;
}
GraphicLayer* LayerRGTRepOnSlice::layer(QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) {
	if (m_layer == nullptr) {
		m_layer = new LayerRGTSliceLayer(this,m_dir,m_transfoProvider,m_currentSlice, scene, defaultZDepth, parent);
	}
	return m_layer;
}

bool LayerRGTRepOnSlice::setSampleUnit(SampleUnit unit) {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return list.contains(unit);
}

QList<SampleUnit> LayerRGTRepOnSlice::getAvailableSampleUnits() const {
	CubeSeismicAddon addon = m_layerslice->seismic()->cubeSeismicAddon();
	QList<SampleUnit> list;
	list.push_back(addon.getSampleUnit());
	return list;
}

QString LayerRGTRepOnSlice::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return (list.contains(sampleUnit)) ? "" : "Dataset unit not compatible";
}

// MZR 16072021
void LayerRGTRepOnSlice::buildContextMenu(QMenu * menu){
	QAction *deleteAction = new QAction(tr("Delete Layers"), this);
	menu->addAction(deleteAction);
	connect(deleteAction, SIGNAL(triggered()), this, SLOT(deleteLayerRGTRepOnSlice()));
}
// MZR 16072021
void LayerRGTRepOnSlice::deleteLayerRGTRepOnSlice(){
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

AbstractGraphicRep::TypeRep LayerRGTRepOnSlice::getTypeGraphicRep() {
    return AbstractGraphicRep::Courbe;
}
