#include "fixedrgblayersfromdatasetandcubereponslice.h"

#include "fixedrgblayersfromdatasetandcubeproppanelonslice.h"
#include "fixedrgblayersfromdatasetandcube.h"
#include "globalconfig.h"
#include "abstractinnerview.h"
#include "fixedrgblayersfromdatasetandcubelayeronslice.h"
#include "cubeseismicaddon.h"
#include <fileInformationWidget.h>
#include <QColorDialog>
#include <QMessageBox>
#include <QMenu>
#include <gccOnSpectrumAttributWidget.h>
#include "seismic3dabstractdataset.h"

FixedRGBLayersFromDatasetAndCubeRepOnSlice::FixedRGBLayersFromDatasetAndCubeRepOnSlice(FixedRGBLayersFromDatasetAndCube *fixedLayer, SliceDirection dir,
		AbstractInnerView *parent) :
		AbstractGraphicRep(parent),ISliceableRep(){
	m_fixedLayer = fixedLayer;
	m_propPanel = nullptr;
	m_layer = nullptr;
	m_dir=dir;
	m_name = m_fixedLayer->name();
	m_currentSlice=0;
}


FixedRGBLayersFromDatasetAndCubeRepOnSlice::~FixedRGBLayersFromDatasetAndCubeRepOnSlice() {
	if (m_layer != nullptr) {
		disconnect(m_fixedLayer, &FixedRGBLayersFromDatasetAndCube::colorChanged, m_layer, &FixedRGBLayersFromDatasetAndCubeLayerOnSlice::refresh);
		delete m_layer;
	}
	if (m_propPanel != nullptr)
		delete m_propPanel;
}

FixedRGBLayersFromDatasetAndCube* FixedRGBLayersFromDatasetAndCubeRepOnSlice::fixedRGBLayersFromDataset() const {
	return m_fixedLayer;
}

void FixedRGBLayersFromDatasetAndCubeRepOnSlice::setSliceIJPosition(int imageVal)
{
	m_currentSlice=imageVal;
	if(m_layer!=nullptr)
		m_layer->setSliceIJPosition(imageVal);
}

IData* FixedRGBLayersFromDatasetAndCubeRepOnSlice::data() const {
	return m_fixedLayer;
}


QWidget* FixedRGBLayersFromDatasetAndCubeRepOnSlice::propertyPanel() {
	if (m_propPanel == nullptr)
	{
		m_propPanel = new FixedRGBLayersFromDatasetAndCubePropPanelOnSlice(this, m_parent);
		connect(m_propPanel, &QWidget::destroyed, [this]() {
			m_propPanel = nullptr;
		});
	}

	return m_propPanel;
}
GraphicLayer* FixedRGBLayersFromDatasetAndCubeRepOnSlice::layer(QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) {
	if (m_layer == nullptr)
	{
		m_fixedLayer->initialize();
		m_layer = new FixedRGBLayersFromDatasetAndCubeLayerOnSlice(this,m_dir,m_currentSlice, scene, defaultZDepth, parent);
		connect(m_fixedLayer, &FixedRGBLayersFromDatasetAndCube::colorChanged, m_layer, &FixedRGBLayersFromDatasetAndCubeLayerOnSlice::refresh);
	}

	return m_layer;
}

bool FixedRGBLayersFromDatasetAndCubeRepOnSlice::setSampleUnit(SampleUnit unit) {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return list.contains(unit);
}

QList<SampleUnit> FixedRGBLayersFromDatasetAndCubeRepOnSlice::getAvailableSampleUnits() const {
	CubeSeismicAddon addon = m_fixedLayer->cubeSeismicAddon();
	QList<SampleUnit> list;
	list.push_back(addon.getSampleUnit());
	return list;
}

QString FixedRGBLayersFromDatasetAndCubeRepOnSlice::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return (list.contains(sampleUnit)) ? "" : "Dataset unit not compatible";
}

AbstractGraphicRep::TypeRep FixedRGBLayersFromDatasetAndCubeRepOnSlice::getTypeGraphicRep() {
    return AbstractGraphicRep::Courbe;
}


void FixedRGBLayersFromDatasetAndCubeRepOnSlice::trt_changeColor()
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

void FixedRGBLayersFromDatasetAndCubeRepOnSlice::trt_properties()
{
	// FileInformationWidget dialog(m_fixedLayer->getIsoFileFromIndex(0));
	// int code = dialog.exec();
	// FileInformationWidget::infoFromFilename(this, m_fixedLayer->getIsoFileFromIndex(0));
	QString info = FileInformationWidget::infoFromFilename(m_fixedLayer->getIsoFileFromIndex(0));
	if ( info.isEmpty() ) return;
	QMessageBox messageBox;
	messageBox.information(m_parent, "Info", info);
}

void FixedRGBLayersFromDatasetAndCubeRepOnSlice::trt_location()
{
	GlobalConfig& config = GlobalConfig::getConfig();
	QString cmd = config.fileExplorerProgram() + " " + m_fixedLayer->dirPath();
	cmd.replace("(", "\\(");
	cmd.replace(")", "\\)");
	system(cmd.toStdString().c_str());
}

void FixedRGBLayersFromDatasetAndCubeRepOnSlice::computeGccOnSpectrum()
{
	qDebug() << m_fixedLayer->name();
	qDebug() << m_fixedLayer->dirPath();
	WorkingSetManager *manager = data()->workingSetManager();
	GccOnSpectrumAttributWidget *p = new GccOnSpectrumAttributWidget(m_fixedLayer->surveyPath(), m_fixedLayer->dirPath(), m_fixedLayer->name(), manager);
	p->show();
}


void FixedRGBLayersFromDatasetAndCubeRepOnSlice::buildContextMenu(QMenu *menu) {
	QAction *attribut = new QAction(tr("Compute gcc on spectrum"), this);
	menu->addAction(attribut);
	connect(attribut, SIGNAL(triggered()), this, SLOT(computeGccOnSpectrum()));
}
