#include "fixedrgblayersfromdatasetandcubereponrandom.h"

#include "fixedrgblayersfromdatasetandcubeproppanelonrandom.h"
#include "fixedrgblayersfromdatasetandcube.h"
#include "globalconfig.h"
#include "abstractinnerview.h"
#include "fixedrgblayersfromdatasetandcubelayeronrandom.h"
#include "cubeseismicaddon.h"
#include "fileInformationWidget.h"
#include "seismic3dabstractdataset.h"

#include <QColorDialog>
#include <QMessageBox>

FixedRGBLayersFromDatasetAndCubeRepOnRandom::FixedRGBLayersFromDatasetAndCubeRepOnRandom(
		FixedRGBLayersFromDatasetAndCube *fixedLayer, AbstractInnerView *parent) :
		AbstractGraphicRep(parent),ISampleDependantRep(){
	m_fixedLayer = fixedLayer;
	m_propPanel = nullptr;
	m_layer = nullptr;
	m_name = m_fixedLayer->name();
}


FixedRGBLayersFromDatasetAndCubeRepOnRandom::~FixedRGBLayersFromDatasetAndCubeRepOnRandom() {
	if (m_layer != nullptr) {
		disconnect(m_fixedLayer, &FixedRGBLayersFromDatasetAndCube::colorChanged, m_layer, &FixedRGBLayersFromDatasetAndCubeLayerOnRandom::refresh);
		delete m_layer;
	}
	if (m_propPanel != nullptr)
		delete m_propPanel;
}

FixedRGBLayersFromDatasetAndCube* FixedRGBLayersFromDatasetAndCubeRepOnRandom::fixedRGBLayersFromDataset() const {
	return m_fixedLayer;
}

IData* FixedRGBLayersFromDatasetAndCubeRepOnRandom::data() const {
	return m_fixedLayer;
}


QWidget* FixedRGBLayersFromDatasetAndCubeRepOnRandom::propertyPanel() {
	if (m_propPanel == nullptr)
	{
		m_propPanel = new FixedRGBLayersFromDatasetAndCubePropPanelOnRandom(this, m_parent);
		connect(m_propPanel, &QWidget::destroyed, [this]() {
			m_propPanel = nullptr;
		});
	}

	return m_propPanel;
}
GraphicLayer* FixedRGBLayersFromDatasetAndCubeRepOnRandom::layer(QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) {
	if (m_layer == nullptr)
	{
		m_fixedLayer->initialize();
		m_layer = new FixedRGBLayersFromDatasetAndCubeLayerOnRandom(this, scene, defaultZDepth, parent);
		connect(m_fixedLayer, &FixedRGBLayersFromDatasetAndCube::colorChanged, m_layer, &FixedRGBLayersFromDatasetAndCubeLayerOnRandom::refresh);
	}
	return m_layer;
}

bool FixedRGBLayersFromDatasetAndCubeRepOnRandom::setSampleUnit(SampleUnit unit) {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return list.contains(unit);
}

QList<SampleUnit> FixedRGBLayersFromDatasetAndCubeRepOnRandom::getAvailableSampleUnits() const {
	CubeSeismicAddon addon = m_fixedLayer->cubeSeismicAddon();
	QList<SampleUnit> list;
	list.push_back(addon.getSampleUnit());
	return list;
}

QString FixedRGBLayersFromDatasetAndCubeRepOnRandom::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return (list.contains(sampleUnit)) ? "" : "Dataset unit not compatible";
}

AbstractGraphicRep::TypeRep FixedRGBLayersFromDatasetAndCubeRepOnRandom::getTypeGraphicRep() {
    return AbstractGraphicRep::Courbe;
}

void FixedRGBLayersFromDatasetAndCubeRepOnRandom::deleteLayer(){
    if (m_layer!=nullptr) {
        delete m_layer;
        m_layer = nullptr;
    }

    if (m_propPanel != nullptr){
        delete m_propPanel;
        m_propPanel = nullptr;
    }
}

void FixedRGBLayersFromDatasetAndCubeRepOnRandom::trt_changeColor()
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

void FixedRGBLayersFromDatasetAndCubeRepOnRandom::trt_properties()
{
	// FileInformationWidget dialog(m_fixedLayer->getIsoFileFromIndex(0));
	// int code = dialog.exec();
	// FileInformationWidget::infoFromFilename(this, m_fixedLayer->getIsoFileFromIndex(0));
	QString info = FileInformationWidget::infoFromFilename(m_fixedLayer->getIsoFileFromIndex(0));
	if ( info.isEmpty() ) return;
	QMessageBox messageBox;
	messageBox.information(m_parent, "Info", info);
}

void FixedRGBLayersFromDatasetAndCubeRepOnRandom::trt_location()
{
	GlobalConfig& config = GlobalConfig::getConfig();
	QString cmd = config.fileExplorerProgram() + " " + m_fixedLayer->dirPath();
	cmd.replace("(", "\\(");
	cmd.replace(")", "\\)");
	system(cmd.toStdString().c_str());
}
