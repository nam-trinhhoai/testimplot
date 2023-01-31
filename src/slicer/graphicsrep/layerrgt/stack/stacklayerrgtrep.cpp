#include "stacklayerrgtrep.h"

#include "cudaimagepaletteholder.h"
#include "stacklayerrgtproppanel.h"
#include "stacklayerrgtlayer.h"
#include "qgllineitem.h"
#include "datacontroler.h"
#include "slicepositioncontroler.h"
#include "LayerSlice.h"
#include "seismic3dabstractdataset.h"
#include "seismic3ddataset.h"
#include "affinetransformation.h"
#include "affine2dtransformation.h"
#include "abstractinnerview.h"
#include "workingsetmanager.h"
#include "rgtqglcudaimageitem.h"
#include <QMenu>
#include <QAction>

StackLayerRGTRep::StackLayerRGTRep(LayerSlice *layerslice, AbstractInnerView *parent) :
		AbstractGraphicRep(parent),  IMouseImageDataProvider(), ISliceableRep() {
	m_layerSlice = layerslice;
	m_propPanel = nullptr;
	m_layer = nullptr;
	m_showCrossHair = false;
	m_name = m_layerSlice->name();
	m_currentStackIndex = 0;

	m_image = new CUDAImagePaletteHolder(m_layerSlice->width(), m_layerSlice->depth(),
			ImageFormats::QSampleType::FLOAT32, m_layerSlice->seismic()->ijToXYTransfo(),
			parent);


	connect(m_image, SIGNAL(dataChanged()), this,
			SLOT(dataChanged()));

	connect(m_layerSlice, &LayerSlice::computationFinished, this, &StackLayerRGTRep::updateNbOutputSlices);
	connect(m_layerSlice,SIGNAL(deletedMenu()),this,SLOT(deleteStackLayerRGTRep()));
}

StackLayerRGTRep::~StackLayerRGTRep() {
	if (m_layer != nullptr)
		delete m_layer;
	if (m_propPanel != nullptr)
		delete m_propPanel;
}

LayerSlice* StackLayerRGTRep::layerSlice() const {
	return m_layerSlice;
}

CUDAImagePaletteHolder* StackLayerRGTRep::isoSurfaceHolder() {
	return m_layerSlice->isoSurfaceHolder();
}

void StackLayerRGTRep::dataChanged() {
	if (m_propPanel != nullptr)
		m_propPanel->updatePalette();
	if (m_layer != nullptr)
		m_layer->refresh();
}

IData* StackLayerRGTRep::data() const {
	return m_layerSlice;
}

QWidget* StackLayerRGTRep::propertyPanel() {
	if (m_propPanel == nullptr) {
		m_propPanel = new StackLayerRGTPropPanel(this,
				m_parent->viewType() == ViewType::View3D, m_parent);
		connect(m_propPanel, &QWidget::destroyed, [this]() {
			m_propPanel = nullptr;
		});
	}
	return m_propPanel;
}
GraphicLayer* StackLayerRGTRep::layer(QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) {
	if (m_layer == nullptr) {
		m_layer = new StackLayerRGTLayer(this, scene, defaultZDepth, parent);
		m_layer->showCrossHair(m_showCrossHair);
	}
	return m_layer;
}

Graphic3DLayer* StackLayerRGTRep::layer3D(QWindow *parent, Qt3DCore::QEntity *root,
		Qt3DRender::QCamera *camera) {
	return nullptr;
}

void StackLayerRGTRep::showCrossHair(bool val) {
	m_showCrossHair = val;
	m_layer->showCrossHair(m_showCrossHair);
}

bool StackLayerRGTRep::crossHair() const {
	return m_showCrossHair;
}

// MZR 16082021
void StackLayerRGTRep::buildContextMenu(QMenu * menu){
	QAction *deleteAction = new QAction(tr("Delete Layers"), this);
	menu->addAction(deleteAction);
	connect(deleteAction, SIGNAL(triggered()), this, SLOT(deleteStackLayerRGTRep()));
}
// MZR 16072021
void StackLayerRGTRep::deleteStackLayerRGTRep(){
	m_parent->hideRep(this);
	emit deletedRep(this);

	// Spectrum layer
	if(m_layerSlice->getComputationType() == 1){
		m_layerSlice->deleteRgt();
	}

	connect(m_layerSlice,nullptr,this,nullptr);
	m_layerSlice->deleteRep();
	this->deleteLater();
}

bool StackLayerRGTRep::mouseData(double x, double y, MouseInfo &info) {
	double value;
	bool valid = IGeorefImage::value(m_image, x, y, info.i,
			info.j, value);
	info.valuesDesc.push_back("Attribute");
	info.values.push_back(value);
	info.depthValue = true;
	IGeorefImage::value(isoSurfaceHolder(), x, y, info.i, info.j,
			value);
	double realDepth;
	m_layerSlice->seismic()->sampleTransformation()->direct(value, realDepth);
	info.depth = realDepth;
	info.depthUnit = m_layerSlice->seismic()->cubeSeismicAddon().getSampleUnit();
	return valid;
}

int StackLayerRGTRep::currentSliceIJPosition() const {
	return m_currentStackIndex;
}

void StackLayerRGTRep::setSliceIJPosition(int position) {
	m_currentStackIndex = position;
	const float* tab = nullptr;
	if ( m_layerSlice && position<m_layerSlice->getNbOutputSlices()) {
		tab = m_layerSlice->getModuleData(position);
	}

	if (!tab)
		return;

//	int d = m_datasetS->depth();
//	int w = m_datasetS->width();
//	int h = m_datasetS->height();
//
//	Seismic3DDataset *seismic = dynamic_cast<Seismic3DDataset*>(m_datasetS);
//	Seismic3DDataset *rgt = dynamic_cast<Seismic3DDataset*>(m_datasetT);

	m_image->updateTexture(tab, false);

	QString label = m_layerSlice->getLabelFromPosition(m_currentStackIndex);
	std::pair<bool, LayerSlice::PaletteParameters> lockedPalette =
			m_layerSlice->getLockedPalette(label);
	if (lockedPalette.first) {
		m_image->setRange(lockedPalette.second.range);
		m_image->setLookupTable(lockedPalette.second.lookupTable);
	}

	emit sliceIJPositionChanged(m_currentStackIndex);
}

void StackLayerRGTRep::updateNbOutputSlices(int nbOutputSlices) {
	QVector2D outVec(0, nbOutputSlices);
	emit stackRangeChanged(outVec);

	setSliceIJPosition(m_currentStackIndex);
}

QVector2D StackLayerRGTRep::stackRange() const {
	QVector2D outVec(0, m_layerSlice->getNbOutputSlices()-1);
	return outVec;
}

QString StackLayerRGTRep::getLabelFromPosition(int val) {
	return m_layerSlice->getLabelFromPosition(val);
}

QString StackLayerRGTRep::getCurrentLabel() {
	return getLabelFromPosition(m_currentStackIndex);
}

AbstractGraphicRep::TypeRep StackLayerRGTRep::getTypeGraphicRep() {
    return AbstractGraphicRep::Image;
}

void StackLayerRGTRep::deleteGraphicItemDataContent(QGraphicsItem *item)
{
	deleteData(m_image,item);
}

QGraphicsObject* StackLayerRGTRep::cloneCUDAImageWithMask(QGraphicsItem *parent)
{
	const float* tab = nullptr, *tabIso = nullptr;
	CUDAImagePaletteHolder *new_img = new CUDAImagePaletteHolder(m_layerSlice->width(), m_layerSlice->depth(),
				ImageFormats::QSampleType::FLOAT32, m_layerSlice->seismic()->ijToXYTransfo(),
				this);

	CUDAImagePaletteHolder *surfaceholder =  new CUDAImagePaletteHolder(m_layerSlice->width(), m_layerSlice->depth(),
			ImageFormats::QSampleType::FLOAT32, m_layerSlice->seismic()->ijToXYTransfo(),
			this);

	tab = m_layerSlice->getModuleData(m_currentStackIndex);

	new_img->updateTexture(tab, false);
	new_img->setRange(m_image->range());

	tabIso = m_layerSlice->getModuleData(0);
	surfaceholder->updateTexture(tabIso, false);
	surfaceholder->setRange(isoSurfaceHolder()->range());

	return new RGTQGLCUDAImageItem(surfaceholder,
			new_img, parent, true);
}

