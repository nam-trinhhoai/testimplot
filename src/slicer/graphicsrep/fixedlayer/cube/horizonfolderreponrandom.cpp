#include "horizonfolderreponrandom.h"

#include "DataUpdatorDialog.h"
#include "globalUtil.h"
#include "horizonAttributComputeDialog.h"
#include "horizonfolderproppanelonrandom.h"
#include "fixedrgblayersfromdatasetandcube.h"
#include "abstractinnerview.h"
#include "horizonfolderlayeronrandom.h"
#include "cubeseismicaddon.h"
#include <fileInformationWidget.h>
#include <QMessageBox>
#include "importsismagehorizondialog.h"
#include "seismic3dabstractdataset.h"
//#include "horizonfolderdata.h"

HorizonFolderRepOnRandom::HorizonFolderRepOnRandom(HorizonFolderData *fixedLayer,
		AbstractInnerView *parent) :
		AbstractGraphicRep(parent),ISliceableRep(){
	m_fixedLayer = fixedLayer;
	m_propPanel = nullptr;
	m_layer = nullptr;
	//m_dir=dir;
	m_name = m_fixedLayer->name();
	m_currentSlice=0;

	/*actionMenuCreate();
	m_itemMenu = new QMenu("Item Menu");
	m_itemMenu->addAction(m_actionColor);
	m_itemMenu->addAction(m_actionProperties);
	m_itemMenu->addAction(m_actionLocation);*/

	m_polylineShape = new GraphEditor_MultiPolyLineShape();
	m_polylineShape->setReadOnly(true);
	m_polylineShape->setDisplayPerimetre(false);
	//m_polylineShape->setToolTip(m_fixedLayer->currentLayer()->name());//fixedLayer->getName());
	//m_polylineShape->setMenu(m_itemMenu);

	connect(m_fixedLayer, SIGNAL(currentChanged()), this,SLOT(refresh()));

}


HorizonFolderRepOnRandom::~HorizonFolderRepOnRandom() {
	if (m_layer != nullptr)
		delete m_layer;
	if (m_propPanel != nullptr)
		delete m_propPanel;
}


void HorizonFolderRepOnRandom::deleteLayer()
{
	if( m_layer!= nullptr)
	{
		m_layer->deleteLater();
		m_layer = nullptr;
	}
}


/*
FixedRGBLayersFromDatasetAndCube* HorizonFolderRepOnSlice::fixedRGBLayersFromDataset() const {
	return m_fixedLayer;
}*/

void HorizonFolderRepOnRandom::buildContextMenu(QMenu *menu) {
	if (m_name == FREE_HORIZON_LABEL ) {
		QAction *addNVHorizonAction = new QAction("Add Nextvision Horizons", this);
		QAction *addSismageHorizonAction = new QAction("Add Sismage Horizons", this);
		QAction *computeFreeHorizonAttibutAction = new QAction("Compute attributs", this);
		menu->addAction(addNVHorizonAction);
		menu->addAction(addSismageHorizonAction);
		menu->addAction(computeFreeHorizonAttibutAction);
		connect(addNVHorizonAction, SIGNAL(triggered()), this, SLOT(addData()));
		connect(addSismageHorizonAction, SIGNAL(triggered()), this, SLOT(addSismageHorizon()));
		connect(computeFreeHorizonAttibutAction, SIGNAL(triggered()), this, SLOT(computeAttributHorizon()));
	}
}

void HorizonFolderRepOnRandom::addData(){
	DataUpdatorDialog *dialog = new DataUpdatorDialog(m_name, m_fixedLayer->workingSetManager(), nullptr);
	dialog->show();
}

void HorizonFolderRepOnRandom::addSismageHorizon()
{
	ImportSismageHorizonDialog *p = new ImportSismageHorizonDialog(m_fixedLayer->workingSetManager());
	p->show();
}


void HorizonFolderRepOnRandom::computeAttributHorizon()
{
	HorizonAttributComputeDialog *p = new HorizonAttributComputeDialog(nullptr);
	p->setProjectManager(m_fixedLayer->workingSetManager()->getManagerWidget());
	p->setWorkingSetManager(m_fixedLayer->workingSetManager());
	p->show();
}

void HorizonFolderRepOnRandom::setSliceIJPosition(int imageVal)
{
	m_currentSlice=imageVal;
	if(m_layer!=nullptr)
		m_layer->setSliceIJPosition(imageVal);
}

IData* HorizonFolderRepOnRandom::data() const {
	return m_fixedLayer;
}

HorizonFolderData* HorizonFolderRepOnRandom::horizonFolderData() const {
	return m_fixedLayer;
}

QWidget* HorizonFolderRepOnRandom::propertyPanel() {
	if (m_propPanel == nullptr)
	{
		m_propPanel = new HorizonFolderPropPanelOnRandom(this, m_parent);
		connect(m_propPanel, &QWidget::destroyed, [this]() {
			m_propPanel = nullptr;
		});
	}

	return m_propPanel;
}
GraphicLayer* HorizonFolderRepOnRandom::layer(QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) {
	if (m_layer == nullptr)
	{
		m_layer = new HorizonFolderLayerOnRandom(this,m_currentSlice, scene, defaultZDepth, parent);
	}

	return m_layer;
}

void HorizonFolderRepOnRandom::refresh()
{
	setBuffer(m_fixedLayer->isoSurfaceHolder());
}


void HorizonFolderRepOnRandom::setBuffer(CPUImagePaletteHolder* isoSurface )
{
	if(m_layer!= nullptr )m_layer->setBuffer(isoSurface);
}

bool HorizonFolderRepOnRandom::setSampleUnit(SampleUnit unit) {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return list.contains(unit);
}

QList<SampleUnit> HorizonFolderRepOnRandom::getAvailableSampleUnits() const {
	CubeSeismicAddon addon = m_fixedLayer->cubeSeismicAddon();
	QList<SampleUnit> list;
	list.push_back(addon.getSampleUnit());
	return list;
}

QString HorizonFolderRepOnRandom::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return (list.contains(sampleUnit)) ? "" : "Dataset unit not compatible";
}

AbstractGraphicRep::TypeRep HorizonFolderRepOnRandom::getTypeGraphicRep() {
    return AbstractGraphicRep::Courbe;
}

