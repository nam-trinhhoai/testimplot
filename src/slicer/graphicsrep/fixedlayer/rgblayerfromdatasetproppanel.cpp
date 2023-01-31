#include "rgblayerfromdatasetproppanel.h"
#include "rgblayerfromdatasetrep.h"
#include "rgblayerfromdataset.h"
#include "abstractinnerview.h"
#include "pickingtask.h"
#include "pointpickingtask.h"
#include "palettewidget.h"
#include "cudargbimage.h"

#include <iostream>

#include <QComboBox>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QToolButton>
#include <QCheckBox>

RgbLayerFromDatasetPropPanel::RgbLayerFromDatasetPropPanel(RgbLayerFromDatasetRep *rep,
		bool for3D,QWidget *parent) : QWidget(parent) {
	m_rep = rep;
	m_pickingTask = nullptr;
	QVBoxLayout *processLayout = new QVBoxLayout(this);
	//processLayout->setMargin(0);
	processLayout->setContentsMargins(0,0,0,0);

	//palette
	m_paletteRed = new PaletteWidget(this);
	processLayout->addWidget(m_paletteRed, 0, Qt::AlignmentFlag::AlignTop);

	m_paletteRed->setLookupTable(m_rep->image()->get(0)->lookupTable());
	m_paletteRed->setOpacity(m_rep->image()->opacity());

	//Connect the image update
	connect(m_paletteRed, SIGNAL(rangeChanged(const QVector2D &)),
			m_rep->image()->get(0), SLOT(setRange(const QVector2D &)));
	connect(m_paletteRed, SIGNAL(opacityChanged(float)),
			m_rep->image(), SLOT(setOpacity(float)));
	connect(m_paletteRed, SIGNAL(lookupTableChanged(const LookupTable &)),
			m_rep->image()->get(0),
			SLOT(setLookupTable(const LookupTable &)));

	connect(m_rep->image()->get(0),
			SIGNAL(rangeChanged(const QVector2D &)), m_paletteRed,
			SLOT(setRange(const QVector2D &)));
	connect(m_rep->image(), SIGNAL(opacityChanged(float)),
			m_paletteRed, SLOT(setOpacity(float)));
	connect(m_rep->image()->get(0),
			SIGNAL(lookupTableChanged(const LookupTable &)), m_paletteRed,
			SLOT(setLookupTable(const LookupTable &)));

	m_paletteRed->setPaletteHolder(m_rep->image()->get(0));

	//palette
	m_paletteGreen = new PaletteWidget(this);
	processLayout->addWidget(m_paletteGreen, 0, Qt::AlignmentFlag::AlignTop);

	m_paletteGreen->setLookupTable(m_rep->image()->get(1)->lookupTable());
	m_paletteGreen->setOpacity(m_rep->image()->opacity());

	//Connect the image update
	connect(m_paletteGreen, SIGNAL(rangeChanged(const QVector2D &)),
			m_rep->image()->get(1), SLOT(setRange(const QVector2D &)));
	connect(m_paletteGreen, SIGNAL(opacityChanged(float)),
			m_rep->image(), SLOT(setOpacity(float)));
	connect(m_paletteGreen, SIGNAL(lookupTableChanged(const LookupTable &)),
			m_rep->image()->get(1),
			SLOT(setLookupTable(const LookupTable &)));

	connect(m_rep->image()->get(1),
			SIGNAL(rangeChanged(const QVector2D &)), m_paletteGreen,
			SLOT(setRange(const QVector2D &)));
	connect(m_rep->image(), SIGNAL(opacityChanged(float)),
			m_paletteGreen, SLOT(setOpacity(float)));
	connect(m_rep->image()->get(1),
			SIGNAL(lookupTableChanged(const LookupTable &)), m_paletteGreen,
			SLOT(setLookupTable(const LookupTable &)));

	m_paletteGreen->setPaletteHolder(m_rep->image()->get(1));

	//palette
	m_paletteBlue = new PaletteWidget(this);
	processLayout->addWidget(m_paletteBlue, 0, Qt::AlignmentFlag::AlignTop);

	m_paletteBlue->setLookupTable(m_rep->image()->get(2)->lookupTable());
	m_paletteBlue->setOpacity(m_rep->image()->opacity());

	//Connect the image update
	connect(m_paletteBlue, SIGNAL(rangeChanged(const QVector2D &)),
			m_rep->image()->get(2), SLOT(setRange(const QVector2D &)));
	connect(m_paletteBlue, SIGNAL(opacityChanged(float)),
			m_rep->image(), SLOT(setOpacity(float)));
	connect(m_paletteBlue, SIGNAL(lookupTableChanged(const LookupTable &)),
			m_rep->image()->get(2),
			SLOT(setLookupTable(const LookupTable &)));

	connect(m_rep->image()->get(2),
			SIGNAL(rangeChanged(const QVector2D &)), m_paletteBlue,
			SLOT(setRange(const QVector2D &)));
	connect(m_rep->image(), SIGNAL(opacityChanged(float)),
			m_paletteBlue, SLOT(setOpacity(float)));
	connect(m_rep->image()->get(2),
			SIGNAL(lookupTableChanged(const LookupTable &)), m_paletteBlue,
			SLOT(setLookupTable(const LookupTable &)));

	m_paletteBlue->setPaletteHolder(m_rep->image()->get(2));

	processLayout->addWidget(createComboBox(), 0, Qt::AlignmentFlag::AlignTop);

	connect(m_rep->fixedLayer(), &RgbLayerFromDataset::newPropertyCreated, this,
			&RgbLayerFromDatasetPropPanel::addNewItemToComboBoxList);
}

RgbLayerFromDatasetPropPanel::~RgbLayerFromDatasetPropPanel() {

}

void RgbLayerFromDatasetPropPanel::updatePalette() {
	m_paletteRed->setPaletteHolder(m_rep->image()->get(0));
	m_paletteGreen->setPaletteHolder(m_rep->image()->get(1));
	m_paletteBlue->setPaletteHolder(m_rep->image()->get(2));
}

void RgbLayerFromDatasetPropPanel::layerRedChanged(QString value) {
	m_rep->chooseRed(value);
}

void RgbLayerFromDatasetPropPanel::layerGreenChanged(QString value) {
	m_rep->chooseGreen(value);
}

void RgbLayerFromDatasetPropPanel::layerBlueChanged(QString value) {
	m_rep->chooseBlue(value);
}

void RgbLayerFromDatasetPropPanel::pick() {
	if (m_pickButton->isChecked()) {
		m_pickingTask = new PointPickingTask(this);
		m_rep->view()->registerPickingTask(m_pickingTask);
	} else {
		m_rep->view()->unregisterPickingTask(m_pickingTask);
		delete m_pickingTask;
		m_pickingTask = nullptr;
	}
}

void RgbLayerFromDatasetPropPanel::pointPicked(double worldX,
		double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys) {
	std::cout << "Picked position:" << worldX << "\t" << worldY << std::endl;
}

QWidget* RgbLayerFromDatasetPropPanel::createComboBox() {
	m_layerBox = new QGroupBox("Layer", this);
	m_layerComboBoxRed = new QComboBox(this);
	QString attribute = m_rep->redName();
	for (QString key : m_rep->fixedLayer()->keys()) {
		m_layerComboBoxRed->addItem(key);
		if (!attribute.isNull() && !attribute.isEmpty() && attribute.compare(key)==0) {
			m_layerComboBoxRed->setCurrentIndex(m_layerComboBoxRed->count()-1);
		}
	}

	connect(m_layerComboBoxRed, SIGNAL(currentTextChanged(QString)), this,
			SLOT(layerRedChanged(QString)));

	QHBoxLayout *hBox = new QHBoxLayout(m_layerBox);
	//hBox->setMargin(0);
	hBox->setContentsMargins(0,0,0,0);
	hBox->addWidget(m_layerComboBoxRed);

	m_layerComboBoxGreen = new QComboBox(this);
	attribute = m_rep->greenName();
	for (QString key : m_rep->fixedLayer()->keys()) {
		m_layerComboBoxGreen->addItem(key);
		if (!attribute.isNull() && !attribute.isEmpty() && attribute.compare(key)==0) {
			m_layerComboBoxGreen->setCurrentIndex(m_layerComboBoxGreen->count()-1);
		}
	}

	connect(m_layerComboBoxGreen, SIGNAL(currentTextChanged(QString)), this,
			SLOT(layerGreenChanged(QString)));

	hBox->addWidget(m_layerComboBoxGreen);

	m_layerComboBoxBlue = new QComboBox(this);
	attribute = m_rep->blueName();
	for (QString key : m_rep->fixedLayer()->keys()) {
		m_layerComboBoxBlue->addItem(key);
		if (!attribute.isNull() && !attribute.isEmpty() && attribute.compare(key)==0) {
			m_layerComboBoxBlue->setCurrentIndex(m_layerComboBoxBlue->count()-1);
		}
	}

	connect(m_layerComboBoxBlue, SIGNAL(currentTextChanged(QString)), this,
			SLOT(layerBlueChanged(QString)));

	hBox->addWidget(m_layerComboBoxBlue);
	return m_layerBox;
}

void RgbLayerFromDatasetPropPanel::updateComboValueRed(QString value) {
	QSignalBlocker b1(m_layerComboBoxRed);
	std::size_t index = 0;
	while (index<m_layerComboBoxRed->count() && value.compare(m_layerComboBoxRed->itemText(index))!=0) {
		index++;
	}
	if (index<m_layerComboBoxRed->count()) {
		m_layerComboBoxRed->setCurrentIndex(index);
	}
}

void RgbLayerFromDatasetPropPanel::updateComboValueGreen(QString value) {
	QSignalBlocker b1(m_layerComboBoxGreen);
	std::size_t index = 0;
	while (index<m_layerComboBoxGreen->count() && value.compare(m_layerComboBoxGreen->itemText(index))!=0) {
		index++;
	}
	if (index<m_layerComboBoxGreen->count()) {
		m_layerComboBoxGreen->setCurrentIndex(index);
	}
}

void RgbLayerFromDatasetPropPanel::updateComboValueBlue(QString value) {
	QSignalBlocker b1(m_layerComboBoxBlue);
	std::size_t index = 0;
	while (index<m_layerComboBoxBlue->count() && value.compare(m_layerComboBoxBlue->itemText(index))!=0) {
		index++;
	}
	if (index<m_layerComboBoxBlue->count()) {
		m_layerComboBoxBlue->setCurrentIndex(index);
	}
}

void RgbLayerFromDatasetPropPanel::addNewItemToComboBoxList(QString item) {
	m_layerComboBoxRed->addItem(item);
	m_layerComboBoxGreen->addItem(item);
	m_layerComboBoxBlue->addItem(item);
}
