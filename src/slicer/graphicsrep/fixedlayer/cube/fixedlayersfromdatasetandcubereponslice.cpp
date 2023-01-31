#include <QColorDialog>
#include <QMessageBox>

#include <fileInformationWidget.h>
#include "fixedlayersfromdatasetandcubereponslice.h"

#include "fixedlayersfromdatasetandcubeproppanelonslice.h"
#include <fixedlayersfreehorizonproppanelonslice.h>
#include "fixedlayersfromdatasetandcube.h"
#include "globalconfig.h"
#include "abstractinnerview.h"
#include "fixedlayersfromdatasetandcubelayeronslice.h"
#include "cubeseismicaddon.h"
#include "seismic3dabstractdataset.h"
#include "cudaimagepaletteholder.h"

FixedLayersFromDatasetAndCubeRepOnSlice::FixedLayersFromDatasetAndCubeRepOnSlice(FixedLayersFromDatasetAndCube *fixedLayer, SliceDirection dir,
		AbstractInnerView *parent) :
		AbstractGraphicRep(parent),ISliceableRep(){
	m_fixedLayer = fixedLayer;
	m_propPanel = nullptr;
	m_layer = nullptr;
	m_dir=dir;
	m_name = m_fixedLayer->name();
	m_currentSlice=0;
}


FixedLayersFromDatasetAndCubeRepOnSlice::~FixedLayersFromDatasetAndCubeRepOnSlice() {
	if (m_layer != nullptr) {
		disconnect(m_fixedLayer, &FixedLayersFromDatasetAndCube::colorChanged, m_layer, &FixedLayersFromDatasetAndCubeLayerOnSlice::refresh);
		delete m_layer;
	}
	if (m_propPanel != nullptr)
		delete m_propPanel;
}

FixedLayersFromDatasetAndCube* FixedLayersFromDatasetAndCubeRepOnSlice::fixedLayersFromDataset() const {
	return m_fixedLayer;
}

void FixedLayersFromDatasetAndCubeRepOnSlice::setSliceIJPosition(int imageVal)
{
	m_currentSlice=imageVal;
	if(m_layer!=nullptr)
		m_layer->setSliceIJPosition(imageVal);
}

IData* FixedLayersFromDatasetAndCubeRepOnSlice::data() const {
	return m_fixedLayer;
}


QWidget* FixedLayersFromDatasetAndCubeRepOnSlice::propertyPanel() {
	if (m_propPanel == nullptr)
	{
		if ( m_fixedLayer->enableSlicePropertyPanel() )
		{
			m_propPanel = new FixedLayersFromDatasetAndCubePropPanelOnSlice(this, m_parent);
			connect(m_propPanel, &QWidget::destroyed, [this]() {
				m_propPanel = nullptr;
			});
		}
		else
		{
			/*
			m_propPanel = new FixedLayersFreeHorizonPropPanelOnSlice(this, m_parent);
			connect(m_propPanel, &QWidget::destroyed, [this]() {
				m_propPanel = nullptr;
			});
			*/
		}
	}

	return m_propPanel;
}
GraphicLayer* FixedLayersFromDatasetAndCubeRepOnSlice::layer(QGraphicsScene *scene, int defaultZDepth,
		QGraphicsItem *parent) {
	if (m_layer == nullptr)
	{
		m_fixedLayer->initialize();
		m_layer = new FixedLayersFromDatasetAndCubeLayerOnSlice(this,m_dir,m_currentSlice, scene, defaultZDepth, parent);

		connect(m_fixedLayer, &FixedLayersFromDatasetAndCube::colorChanged, m_layer, &FixedLayersFromDatasetAndCubeLayerOnSlice::refresh);
	}
	return m_layer;
}

bool FixedLayersFromDatasetAndCubeRepOnSlice::setSampleUnit(SampleUnit unit) {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return list.contains(unit);
}

QList<SampleUnit> FixedLayersFromDatasetAndCubeRepOnSlice::getAvailableSampleUnits() const {
	CubeSeismicAddon addon = m_fixedLayer->cubeSeismicAddon();
	QList<SampleUnit> list;
	list.push_back(addon.getSampleUnit());
	return list;
}

QString FixedLayersFromDatasetAndCubeRepOnSlice::getSampleUnitErrorMessage(SampleUnit sampleUnit) const {
	QList<SampleUnit> list = getAvailableSampleUnits();
	return (list.contains(sampleUnit)) ? "" : "Dataset unit not compatible";
}

AbstractGraphicRep::TypeRep FixedLayersFromDatasetAndCubeRepOnSlice::getTypeGraphicRep() {
	return AbstractGraphicRep::Courbe;
}

void FixedLayersFromDatasetAndCubeRepOnSlice::trt_changeColor()
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

void FixedLayersFromDatasetAndCubeRepOnSlice::trt_properties()
{
	// FileInformationWidget dialog(m_fixedLayer->getIsoFileFromIndex(0));
	// int code = dialog.exec();
	// FileInformationWidget::infoFromFilename(this, m_fixedLayer->getIsoFileFromIndex(0));
	QString info = FileInformationWidget::infoFromFilename(m_fixedLayer->getIsoFileFromIndex(0));
	if ( info.isEmpty() ) return;
	QMessageBox messageBox;
	messageBox.information(m_parent, "Info", info);
}


void FixedLayersFromDatasetAndCubeRepOnSlice::trt_location()
{
	GlobalConfig& config = GlobalConfig::getConfig();
	QString cmd = config.fileExplorerProgram() + " " + m_fixedLayer->dirPath();
	cmd.replace("(", "\\(");
	cmd.replace(")", "\\)");
	system(cmd.toStdString().c_str());
}
