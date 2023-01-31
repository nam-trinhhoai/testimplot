#include "horizonfolderreponslice.h"

#include "DataUpdatorDialog.h"
#include "globalUtil.h"
#include "horizonAttributComputeDialog.h"
#include "horizonfolderproppanelonslice.h"
#include "fixedrgblayersfromdatasetandcube.h"
#include "abstractinnerview.h"
#include "horizonfolderlayeronslice.h"
#include "cubeseismicaddon.h"
#include <fileInformationWidget.h>
#include <QMessageBox>
#include "importsismagehorizondialog.h"
#include "seismic3dabstractdataset.h"
//#include "horizonfolderdata.h"

HorizonFolderRepOnSlice::HorizonFolderRepOnSlice(HorizonFolderData *fixedLayer, SliceDirection dir,
		AbstractInnerView *parent) :
		AbstractGraphicRep(parent),ISliceableRep(){
	m_fixedLayer = fixedLayer;
	m_propPanel = nullptr;
	m_layer = nullptr;
	m_dir=dir;
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


HorizonFolderRepOnSlice::~HorizonFolderRepOnSlice() {
	if (m_layer != nullptr)
		delete m_layer;
	if (m_propPanel != nullptr)
		delete m_propPanel;
}
/*
FixedRGBLayersFromDatasetAndCube* HorizonFolderRepOnSlice::fixedRGBLayersFromDataset() const {
	return m_fixedLayer;
}*/

void HorizonFolderRepOnSlice::buildContextMenu(QMenu *menu) {
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

void HorizonFolderRepOnSlice::addData(){
	DataUpdatorDialog *dialog = new DataUpdatorDialog(m_name, m_fixedLayer->workingSetManager(), nullptr);
	dialog->show();
}

void HorizonFolderRepOnSlice::addSismageHorizon()
{
	ImportSismageHorizonDialog *p = new ImportSismageHorizonDialog(m_fixedLayer->workingSetManager());
	p->show();
}


void HorizonFolderRepOnSlice::computeAttributHorizon()
{
	HorizonAttributComputeDialog *p = new HorizonAttributComputeDialog(nullptr);
	p->setProjectManager(m_fixedLayer->workingSetManager()->getManagerWidget());
	p->setWorkingSetManager(m_fixedLayer->workingSetManager());
	p->show();
}

void HorizonFolderRepOnSlice::setSliceIJPosition(int imageVal)
{
	m_currentSlice=imageVal;
	if(m_layer!=nullptr)
		m_layer->setSliceIJPosition(imageVal);
}

IData* HorizonFolderRepOnSlice::data() const {
	return m_fixedLayer;
}

HorizonFolderData* HorizonFolderRepOnSlice::horizonFolderData() const {
	return m_fixedLayer;
}

QWidget* HorizonFolderRepOnSlice::propertyPanel() {
	if (m_propPanel == nullptr)
	{
		m_propPanel = new HorizonFolderPropPanelOnSlice(this, m_parent);
		connect(m_propPanel, &QWidget::destroyed, [this]() {
			m_propPanel = nullptr;
		});
	}

	return m_propPanel;
}
GraphicLayer* HorizonFolderRepOnSlice::layer(QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) {
	if (m_layer == nullptr)
	{
		m_layer = new HorizonFolderLayerOnSlice(this,m_dir,m_currentSlice, scene, defaultZDepth, parent);
	}

	return m_layer;
}

void HorizonFolderRepOnSlice::refresh()
{
	setBuffer(m_fixedLayer->isoSurfaceHolder());
}


void HorizonFolderRepOnSlice::setBuffer(CPUImagePaletteHolder* isoSurface )
{
	if(m_layer!= nullptr )m_layer->setBuffer(isoSurface);
}

bool HorizonFolderRepOnSlice::setSampleUnit(SampleUnit unit) {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return list.contains(unit);
}

QList<SampleUnit> HorizonFolderRepOnSlice::getAvailableSampleUnits() const {
	CubeSeismicAddon addon = m_fixedLayer->cubeSeismicAddon();
	QList<SampleUnit> list;
	list.push_back(addon.getSampleUnit());
	return list;
}

QString HorizonFolderRepOnSlice::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return (list.contains(sampleUnit)) ? "" : "Dataset unit not compatible";
}

AbstractGraphicRep::TypeRep HorizonFolderRepOnSlice::getTypeGraphicRep() {
    return AbstractGraphicRep::Courbe;
}

/*
void HorizonFolderRepOnSlice::trt_changeColor()
{
	QColorDialog dialog;
	dialog.setCurrentColor(m_fixedLayer->getHorizonColor());
	dialog.setOption (QColorDialog::DontUseNativeDialog);
	if (dialog.exec() == QColorDialog::Accepted)
	{
		QColor color = dialog.currentColor();
		m_fixedLayer->setHorizonColor(color);
		// item->setTextColor(0, color);
		m_layer->refresh();
		m_fixedLayer->updateTextColor(m_fixedLayer->getName());
	}
}

void HorizonFolderRepOnSlice::trt_properties()
{
	// FileInformationWidget dialog(m_fixedLayer->getIsoFileFromIndex(0));
	// int code = dialog.exec();
	// FileInformationWidget::infoFromFilename(this, m_fixedLayer->getIsoFileFromIndex(0));
	QString info = FileInformationWidget::infoFromFilename(m_fixedLayer->getIsoFileFromIndex(0));
	if ( info.isEmpty() ) return;
	QMessageBox messageBox;
	messageBox.information(m_parent, "Info", info);
}

void HorizonFolderRepOnSlice::trt_location()
{
	QString cmd = "caja " + m_fixedLayer->dirPath();
	cmd.replace("(", "\\(");
	cmd.replace(")", "\\)");
	system(cmd.toStdString().c_str());
}*/

/*
void HorizonFolderRepOnSlice::actionMenuCreate()
{
	m_actionColor = new QAction(QIcon(":/slicer/icons/graphic_tools/paint_bucket.png"), tr("color"), this);
	connect(m_actionColor, &QAction::triggered, this, &HorizonFolderRepOnSlice::trt_changeColor);
	m_actionProperties = new QAction(QIcon(":/slicer/icons/graphic_tools/info.png"), tr("properties"), this);
	connect(m_actionProperties, &QAction::triggered, this, &HorizonFolderRepOnSlice::trt_properties);
	m_actionLocation = new QAction(QIcon(":/slicer/icons/graphic_tools/info.png"), tr("folder"), this);
	connect(m_actionLocation, &QAction::triggered, this, &HorizonFolderRepOnSlice::trt_location);
}*/
