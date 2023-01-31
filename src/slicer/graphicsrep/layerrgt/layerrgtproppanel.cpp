#include "layerrgtproppanel.h"

#include <iostream>
#include "layerrgtrep.h"
#include "palettewidget.h"
#include "cudaimagepaletteholder.h"
#include "LayerSlice.h"

#include <QGroupBox>
#include <QHBoxLayout>
#include <QSlider>
#include <QSpinBox>
#include <QToolButton>
#include <QAction>
#include <QLabel>
#include <QLineEdit>
#include <QCheckBox>
#include "abstractinnerview.h"
#include "pointpickingtask.h"

LayerRGTPropPanel::LayerRGTPropPanel(LayerRGTRep *rep, bool for3D, QWidget *parent) :
		QWidget(parent) {
	m_rep = rep;
	m_pickingTask = nullptr;
	m_nameAttribut =m_rep->layerSlice()->getLabelFromPosition(0);
	QVBoxLayout *processLayout = new QVBoxLayout(this);
	//processLayout->setMargin(0);
	processLayout->setContentsMargins(0,0,0,0);

	//palette
	m_palette = new PaletteWidget(this);
	processLayout->addWidget(m_palette, 0, Qt::AlignmentFlag::AlignTop);

	m_palette->setLookupTable(m_rep->layerSlice()->image()->lookupTable());
	m_palette->setOpacity(m_rep->layerSlice()->image()->opacity());

	//Connect the image update
	connect(m_palette, SIGNAL(rangeChanged(const QVector2D &)),
			m_rep->layerSlice()->image(), SLOT(setRange(const QVector2D &)));
	connect(m_palette, SIGNAL(opacityChanged(float)),
			m_rep->layerSlice()->image(), SLOT(setOpacity(float)));
	connect(m_palette, SIGNAL(lookupTableChanged(const LookupTable &)),
			m_rep->layerSlice()->image(),
			SLOT(setLookupTable(const LookupTable &)));

	connect(m_rep->layerSlice()->image(),
			SIGNAL(rangeChanged(const QVector2D &)), m_palette,
			SLOT(setRange(const QVector2D &)));
	connect(m_rep->layerSlice()->image(), SIGNAL(opacityChanged(float)),
			m_palette, SLOT(setOpacity(float)));
	connect(m_rep->layerSlice()->image(),
			SIGNAL(lookupTableChanged(const LookupTable &)), m_palette,
			SLOT(setLookupTable(const LookupTable &)));

	m_palette->setPaletteHolder(m_rep->layerSlice()->image());

	processLayout->addWidget(createSliceBox(), 0, Qt::AlignmentFlag::AlignTop);

	//Window
	QWidget *rangeWidget = new QWidget(this);
	QHBoxLayout *hBox = new QHBoxLayout(rangeWidget);
	//hBox->setMargin(0);
	hBox->setContentsMargins(0,0,0,0);
	m_window = new QLineEdit();
	m_window->setLocale(QLocale::C);
	connect(m_window, SIGNAL(returnPressed()), this, SLOT(valueChanged()));
	//TODO a nettoyer entierement!!!
	//hBox->addWidget(new QLabel("RMS Extraction Window:"));
	//hBox->addWidget(m_window);


	processLayout->addWidget(rangeWidget, 0, Qt::AlignmentFlag::AlignTop);
	if (!for3D) {
		m_showCrossHair = new QCheckBox("Show crosshair", this);
		m_showCrossHair->setChecked(m_rep->crossHair());
		connect(m_showCrossHair, SIGNAL(stateChanged(int)), this,
				SLOT(showCrossHair(int)));

		processLayout->addWidget(m_showCrossHair, 0,
				Qt::AlignmentFlag::AlignTop);
	}
	{
		QSignalBlocker b1(m_sliceSpin);
		QSignalBlocker b2(m_sliceSlider);
		m_sliceSlider->setMaximum(m_rep->layerSlice()->getNbOutputSlices()-1);
		m_sliceSlider->setMinimum(0);
		m_sliceSlider->setValue(m_rep->layerSlice()->currentPosition());

		m_sliceSpin->setMaximum(m_rep->layerSlice()->getNbOutputSlices()-1);
		m_sliceSpin->setMinimum(0);
		m_sliceSpin->setValue(m_rep->layerSlice()->currentPosition());
	}

	connect(m_rep->layerSlice(), SIGNAL(extractionWindowChanged(unsigned int)),
			this, SLOT(extractionWindowChanged(unsigned int)));
	connect(m_rep->layerSlice(), SIGNAL(RGTIsoValueChanged(int)), this,
			SLOT(RGTIsoValueChanged(int)));
	connect(m_rep->layerSlice(), &LayerSlice::computationFinished, [this]() {
		if (m_rep->layerSlice()->currentPosition()!=m_sliceSpin->value()) {
			QSignalBlocker b1(m_sliceSpin);
			QSignalBlocker b2(m_sliceSlider);
			m_sliceSlider->setValue(m_rep->layerSlice()->currentPosition());

			m_sliceSpin->setValue(m_rep->layerSlice()->currentPosition());
		}
		if (m_rep->layerSlice()->getNbOutputSlices()!=m_sliceSpin->maximum()+1) {
			QSignalBlocker b1(m_sliceSpin);
			QSignalBlocker b2(m_sliceSlider);
			m_sliceSlider->setMaximum(m_rep->layerSlice()->getNbOutputSlices()-1);

			m_sliceSpin->setMaximum(m_rep->layerSlice()->getNbOutputSlices()-1);
		}
	});

	m_lockPalette = new QCheckBox("Lock Palette", this);
	updateLockCheckBox();
	connect(m_lockPalette, SIGNAL(stateChanged(int)), this,
				SLOT(lockPalette(int)));

	processLayout->addWidget(m_lockPalette, 0,
				Qt::AlignmentFlag::AlignTop);

	connect(m_palette, SIGNAL(rangeChanged(const QVector2D &)),
			this, SLOT(updateLockRange(const QVector2D &)));
	connect(m_palette, SIGNAL(lookupTableChanged(const LookupTable &)),
			this, SLOT(updateLockLookupTable(const LookupTable &)));

	connect(m_rep->layerSlice()->image(),
			SIGNAL(rangeChanged(const QVector2D &)), this,
			SLOT(updateLockRange(const QVector2D &)));
	connect(m_rep->layerSlice()->image(),
			SIGNAL(lookupTableChanged(const LookupTable &)), this,
			SLOT(updateLockLookupTable(const LookupTable &)));

	extractionWindowChanged(m_rep->layerSlice()->extractionWindow());
	RGTIsoValueChanged(m_rep->layerSlice()->currentPosition());
}
LayerRGTPropPanel::~LayerRGTPropPanel() {

}

void LayerRGTPropPanel::pick() {
	if (m_pickButton->isChecked()) {
		m_pickingTask = new PointPickingTask(this);
		m_rep->view()->registerPickingTask(m_pickingTask);
	} else {
		m_rep->view()->unregisterPickingTask(m_pickingTask);
		delete m_pickingTask;
		m_pickingTask = nullptr;
	}
}

void LayerRGTPropPanel::pointPicked(double worldX, double worldY,
		Qt::MouseButton button, Qt::KeyboardModifiers keys) {
	std::cout << "Picked position:" << worldX << "\t" << worldY << std::endl;

}

void LayerRGTPropPanel::extractionWindowChanged(unsigned int size) {
	QSignalBlocker b(m_window);
	m_window->setText(
			locale().toString(m_rep->layerSlice()->extractionWindow()));
}
void LayerRGTPropPanel::RGTIsoValueChanged(int pos) {
	updateSpinValue(pos, m_sliceSlider, m_sliceSpin);
	updateLockCheckBox();
}
void LayerRGTPropPanel::updateSpinValue(int value, QSlider *slider, QSpinBox *spin) {
	QSignalBlocker b1(slider);
	QSignalBlocker b2(spin);
	slider->setValue(value);
	spin->setValue(value);
}

void LayerRGTPropPanel::showCrossHair(int value) {
	m_rep->showCrossHair(value == Qt::Checked);
}
QWidget* LayerRGTPropPanel::createSliceBox() {
	m_sliderBox = new QGroupBox(m_nameAttribut, this);
	m_sliceSlider = new QSlider(Qt::Orientation::Horizontal, this);
	m_sliceSlider->setSingleStep(1);
	m_sliceSlider->setTickInterval(10);
	m_sliceSlider->setMinimum(0);
	m_sliceSlider->setMaximum(1);
	m_sliceSlider->setValue(0);

	m_sliceSpin = new QSpinBox();
	m_sliceSpin->setMinimum(0);
	m_sliceSpin->setMaximum(10000);
	m_sliceSpin->setSingleStep(1);
	m_sliceSpin->setValue(0);

	m_sliceSpin->setWrapping(false);

	connect(m_sliceSpin, SIGNAL(valueChanged(int)), this,
			SLOT(sliceChanged(int)));
	connect(m_sliceSlider, SIGNAL(valueChanged(int)), this,
			SLOT(sliceChanged(int)));

	QHBoxLayout *hBox = new QHBoxLayout(m_sliderBox);
	//hBox->setMargin(0);
	hBox->setContentsMargins(0,0,0,0);
	hBox->addWidget(m_sliceSlider);
	hBox->addWidget(m_sliceSpin);
	return m_sliderBox;
}


void LayerRGTPropPanel::setNameAttribut(QString name)
{
	m_sliderBox->setTitle(name);
}

uint LayerRGTPropPanel::getExtactionWindow() {
	bool ok;
	uint win = locale().toUInt(m_window->text(), &ok);
	if (!ok)
		return m_rep->layerSlice()->extractionWindow();
	return win;
}

void LayerRGTPropPanel::valueChanged() {
	uint win = getExtactionWindow();
	m_rep->layerSlice()->setExtractionWindow(win);
}

void LayerRGTPropPanel::sliceChanged(int value) {
	m_rep->layerSlice()->setSlicePosition(value);
	setNameAttribut(m_rep->layerSlice()->getLabelFromPosition(value));

}

void LayerRGTPropPanel::updatePalette() {
	m_palette->setPaletteHolder(m_rep->layerSlice()->image());
}

void LayerRGTPropPanel::updateLockCheckBox() {
	QString label = m_rep->layerSlice()->getCurrentLabel();
	const std::pair<bool, LayerSlice::PaletteParameters>& lockedPalette =
			m_rep->layerSlice()->getLockedPalette(label);
	int lockState = (lockedPalette.first) ? Qt::Checked : Qt::Unchecked;

	QSignalBlocker b1(m_lockPalette);
	m_lockPalette->setChecked(lockState);
}

void LayerRGTPropPanel::lockPalette(int state) {
	if (state==Qt::Checked) {
		LayerSlice::PaletteParameters params;
		params.range = m_rep->layerSlice()->image()->range();
		params.lookupTable = m_rep->layerSlice()->image()->lookupTable();
		m_rep->layerSlice()->lockPalette(m_rep->layerSlice()->getCurrentLabel(), params);
	} else {
		m_rep->layerSlice()->unlockPalette(m_rep->layerSlice()->getCurrentLabel());
	}
}

void LayerRGTPropPanel::updateLockRange(const QVector2D & range) {
	QString label = m_rep->layerSlice()->getCurrentLabel();
	std::pair<bool, LayerSlice::PaletteParameters> lockedPalette =
		m_rep->layerSlice()->getLockedPalette(label);
	if (lockedPalette.first) {
		LayerSlice::PaletteParameters params = lockedPalette.second;
		params.range = range;
		m_rep->layerSlice()->lockPalette(label, params);
	}
}

void LayerRGTPropPanel::updateLockLookupTable(const LookupTable& lookupTable) {
	QString label = m_rep->layerSlice()->getCurrentLabel();
	std::pair<bool, LayerSlice::PaletteParameters> lockedPalette =
		m_rep->layerSlice()->getLockedPalette(label);
	if (lockedPalette.first) {
		LayerSlice::PaletteParameters params = lockedPalette.second;
		params.lookupTable = lookupTable;
		m_rep->layerSlice()->lockPalette(label, params);
	}
}

