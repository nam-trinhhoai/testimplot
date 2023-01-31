#include "fixedlayersfromdatasetandcubereponrandom.h"

#include "fileInformationWidget.h"
#include "fixedlayersfromdatasetandcubeproppanelonrandom.h"
#include "fixedlayersfromdatasetandcube.h"
#include "globalconfig.h"
#include "abstractinnerview.h"
#include "fixedlayersfromdatasetandcubelayeronrandom.h"
#include "cubeseismicaddon.h"
#include "seismic3dabstractdataset.h"
#include "cpuimagepaletteholder.h"

#include <QColorDialog>
#include <QMessageBox>

FixedLayersFromDatasetAndCubeRepOnRandom::FixedLayersFromDatasetAndCubeRepOnRandom(
		FixedLayersFromDatasetAndCube *fixedLayer, AbstractInnerView *parent) :
		AbstractGraphicRep(parent),ISampleDependantRep(){
	m_fixedLayer = fixedLayer;
	m_propPanel = nullptr;
	m_layer = nullptr;
	m_name = m_fixedLayer->name();
}

FixedLayersFromDatasetAndCubeRepOnRandom::~FixedLayersFromDatasetAndCubeRepOnRandom() {
	if (m_layer != nullptr) {
		disconnect(m_fixedLayer, &FixedLayersFromDatasetAndCube::colorChanged, m_layer, &FixedLayersFromDatasetAndCubeLayerOnRandom::refresh);
		delete m_layer;
	}
	if (m_propPanel != nullptr)
		delete m_propPanel;
}

FixedLayersFromDatasetAndCube* FixedLayersFromDatasetAndCubeRepOnRandom::fixedLayersFromDataset() const {
	return m_fixedLayer;
}

IData* FixedLayersFromDatasetAndCubeRepOnRandom::data() const {
	return m_fixedLayer;
}


QWidget* FixedLayersFromDatasetAndCubeRepOnRandom::propertyPanel() {
	if (m_propPanel == nullptr)
	{
		m_propPanel = new FixedLayersFromDatasetAndCubePropPanelOnRandom(this, m_parent);
		connect(m_propPanel, &QWidget::destroyed, [this]() {
			m_propPanel = nullptr;
		});
	}

	return m_propPanel;
}
GraphicLayer* FixedLayersFromDatasetAndCubeRepOnRandom::layer(QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) {
	if (m_layer == nullptr)
	{
		m_fixedLayer->initialize();
		m_layer = new FixedLayersFromDatasetAndCubeLayerOnRandom(this, scene, defaultZDepth, parent);
		connect(m_fixedLayer, &FixedLayersFromDatasetAndCube::colorChanged, m_layer, &FixedLayersFromDatasetAndCubeLayerOnRandom::refresh);
	}
	return m_layer;
}

bool FixedLayersFromDatasetAndCubeRepOnRandom::setSampleUnit(SampleUnit unit) {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return list.contains(unit);
}

QList<SampleUnit> FixedLayersFromDatasetAndCubeRepOnRandom::getAvailableSampleUnits() const {
	CubeSeismicAddon addon = m_fixedLayer->cubeSeismicAddon();
	QList<SampleUnit> list;
	list.push_back(addon.getSampleUnit());
	return list;
}

QString FixedLayersFromDatasetAndCubeRepOnRandom::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return (list.contains(sampleUnit)) ? "" : "Dataset unit not compatible";
}

AbstractGraphicRep::TypeRep FixedLayersFromDatasetAndCubeRepOnRandom::getTypeGraphicRep() {
	return AbstractGraphicRep::Courbe;
}

void FixedLayersFromDatasetAndCubeRepOnRandom::deleteLayer(){
    if (m_layer != nullptr) {
        delete m_layer;
        m_layer = nullptr;
    }

    if (m_propPanel != nullptr){
        delete m_propPanel;
        m_propPanel = nullptr;
    }
}

void FixedLayersFromDatasetAndCubeRepOnRandom::trt_changeColor()
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
		// m_fixedLayer->updateTextColor(m_fixedLayer->getName());
	}
}

void FixedLayersFromDatasetAndCubeRepOnRandom::trt_properties()
{
	// FileInformationWidget dialog(m_fixedLayer->getIsoFileFromIndex(0));
	// int code = dialog.exec();
	// FileInformationWidget::infoFromFilename(this, m_fixedLayer->getIsoFileFromIndex(0));
	QString info = FileInformationWidget::infoFromFilename(m_fixedLayer->getIsoFileFromIndex(0));
	if ( info.isEmpty() ) return;
	QMessageBox messageBox;
	messageBox.information(m_parent, "Info", info);
}


void FixedLayersFromDatasetAndCubeRepOnRandom::trt_location()
{
	GlobalConfig& config = GlobalConfig::getConfig();
	QString cmd = config.fileExplorerProgram() + " " + m_fixedLayer->dirPath();
	cmd.replace("(", "\\(");
	cmd.replace(")", "\\)");
	system(cmd.toStdString().c_str());
}
