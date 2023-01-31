#include "fixedlayerfromdatasetproppanel.h"
#include "fixedlayerfromdatasetrep.h"
#include "fixedlayerfromdataset.h"
#include "abstractinnerview.h"
#include "pickingtask.h"
#include "pointpickingtask.h"
#include "palettewidget.h"

#include <iostream>

#include <QComboBox>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QToolButton>
#include <QCheckBox>

FixedLayerFromDatasetPropPanel::FixedLayerFromDatasetPropPanel(FixedLayerFromDatasetRep *rep,
		bool for3D,QWidget *parent) : QWidget(parent) {
	m_rep = rep;
	m_pickingTask = nullptr;
	QVBoxLayout *processLayout = new QVBoxLayout(this);
	//processLayout->setMargin(0);
	processLayout->setContentsMargins(0,0,0,0);

	//palette
	m_palette = new PaletteWidget(this);
	processLayout->addWidget(m_palette, 0, Qt::AlignmentFlag::AlignTop);

	m_palette->setLookupTable(m_rep->image()->lookupTable());
	m_palette->setOpacity(m_rep->image()->opacity());

	//Connect the image update
	connect(m_palette, SIGNAL(rangeChanged(const QVector2D &)),
			m_rep->image(), SLOT(setRange(const QVector2D &)));
	connect(m_palette, SIGNAL(opacityChanged(float)),
			m_rep->image(), SLOT(setOpacity(float)));
	connect(m_palette, SIGNAL(lookupTableChanged(const LookupTable &)),
			m_rep->image(),
			SLOT(setLookupTable(const LookupTable &)));

	connect(m_rep->image(),
			SIGNAL(rangeChanged(const QVector2D &)), m_palette,
			SLOT(setRange(const QVector2D &)));
	connect(m_rep->image(), SIGNAL(opacityChanged(float)),
			m_palette, SLOT(setOpacity(float)));
	connect(m_rep->image(),
			SIGNAL(lookupTableChanged(const LookupTable &)), m_palette,
			SLOT(setLookupTable(const LookupTable &)));

	m_palette->setPaletteHolder(m_rep->image());

	processLayout->addWidget(createComboBox(), 0, Qt::AlignmentFlag::AlignTop);


	if (!for3D) {
		m_showCrossHair = new QCheckBox("Show crosshair", this);
		m_showCrossHair->setChecked(m_rep->crossHair());
		connect(m_showCrossHair, SIGNAL(stateChanged(int)), this,
				SLOT(showCrossHair(int)));

		processLayout->addWidget(m_showCrossHair, 0,
				Qt::AlignmentFlag::AlignTop);
	}

	connect(m_rep->fixedLayer(), &FixedLayerFromDataset::newPropertyCreated, this,
			&FixedLayerFromDatasetPropPanel::addNewItemToComboBoxList);
}

FixedLayerFromDatasetPropPanel::~FixedLayerFromDatasetPropPanel() {

}

void FixedLayerFromDatasetPropPanel::updatePalette() {
	m_palette->setPaletteHolder(m_rep->image());
}

void FixedLayerFromDatasetPropPanel::layerChanged(QString value) {
	m_rep->chooseAttribute(value);
}

void FixedLayerFromDatasetPropPanel::showCrossHair(int value) {
	m_rep->showCrossHair(value == Qt::Checked);
}

void FixedLayerFromDatasetPropPanel::pick() {
	if (m_pickButton->isChecked()) {
		m_pickingTask = new PointPickingTask(this);
		m_rep->view()->registerPickingTask(m_pickingTask);
	} else {
		m_rep->view()->unregisterPickingTask(m_pickingTask);
		delete m_pickingTask;
		m_pickingTask = nullptr;
	}
}

void FixedLayerFromDatasetPropPanel::pointPicked(double worldX,
		double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys) {
	std::cout << "Picked position:" << worldX << "\t" << worldY << std::endl;
}

QWidget* FixedLayerFromDatasetPropPanel::createComboBox() {
	m_layerBox = new QGroupBox("Layer", this);
	m_layerComboBox = new QComboBox(this);
	QString attribute = m_rep->attributeName();
	for (QString key : m_rep->fixedLayer()->keys()) {
		m_layerComboBox->addItem(key);
		if (!attribute.isNull() && !attribute.isEmpty() && attribute.compare(key)==0) {
			m_layerComboBox->setCurrentIndex(m_layerComboBox->count()-1);
		}
	}


	connect(m_layerComboBox, SIGNAL(currentTextChanged(QString)), this,
			SLOT(layerChanged(QString)));

	QHBoxLayout *hBox = new QHBoxLayout(m_layerBox);
	//hBox->setMargin(0);
	hBox->setContentsMargins(0,0,0,0);
	hBox->addWidget(m_layerComboBox);
	return m_layerBox;
}

void FixedLayerFromDatasetPropPanel::updateComboValue(QString value) {
	QSignalBlocker b1(m_layerComboBox);
	std::size_t index = 0;
	while (index<m_layerComboBox->count() && value.compare(m_layerComboBox->itemText(index))!=0) {
		index++;
	}
	if (index<m_layerComboBox->count()) {
		m_layerComboBox->setCurrentIndex(index);
	}
}

void FixedLayerFromDatasetPropPanel::addNewItemToComboBoxList(QString item) {
	m_layerComboBox->addItem(item);
}
