#include "stratisliceamplitudeproppanel.h"
#include <iostream>
#include "stratisliceamplituderep.h"

#include "palettewidget.h"
#include "cudaimagepaletteholder.h"
#include "stratislice.h"

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
#include "amplitudestratisliceattribute.h"
#include "stratislice.h"


StratiSliceAmplitudePropPanel::StratiSliceAmplitudePropPanel(StratiSliceAmplitudeRep *rep, bool for3D, QWidget *parent) :
		QWidget(parent) {
	m_rep = rep;
	QVBoxLayout *processLayout = new QVBoxLayout(this);
	//processLayout->setMargin(0);
	processLayout->setContentsMargins(0,0,0,0);

	//palette
	m_palette = new PaletteWidget(this);
	processLayout->addWidget(m_palette, 0, Qt::AlignmentFlag::AlignTop);

	m_palette->setLookupTable(m_rep->stratiSliceAttribute()->image()->lookupTable());
	m_palette->setOpacity(m_rep->stratiSliceAttribute()->image()->opacity());

	//Connect the image update
	connect(m_palette, SIGNAL(rangeChanged(const QVector2D &)),
			m_rep->stratiSliceAttribute()->image(), SLOT(setRange(const QVector2D &)));
	connect(m_palette, SIGNAL(opacityChanged(float)),
			m_rep->stratiSliceAttribute()->image(), SLOT(setOpacity(float)));
	connect(m_palette, SIGNAL(lookupTableChanged(const LookupTable &)),
			m_rep->stratiSliceAttribute()->image(),
			SLOT(setLookupTable(const LookupTable &)));

	connect(m_rep->stratiSliceAttribute()->image(),
			SIGNAL(rangeChanged(const QVector2D &)), m_palette,
			SLOT(setRange(const QVector2D &)));
	connect(m_rep->stratiSliceAttribute()->image(), SIGNAL(opacityChanged(float)),
			m_palette, SLOT(setOpacity(float)));
	connect(m_rep->stratiSliceAttribute()->image(),
			SIGNAL(lookupTableChanged(const LookupTable &)), m_palette,
			SLOT(setLookupTable(const LookupTable &)));

	m_palette->setPaletteHolder(m_rep->stratiSliceAttribute()->image());

	processLayout->addWidget(createSliceBox(), 0, Qt::AlignmentFlag::AlignTop);

	//Window
	QWidget *rangeWidget = new QWidget(this);
	QHBoxLayout *hBox = new QHBoxLayout(rangeWidget);
	//hBox->setMargin(0);
	hBox->setContentsMargins(0,0,0,0);
	m_window = new QLineEdit();
	m_window->setLocale(QLocale::C);
	connect(m_window, SIGNAL(returnPressed()), this, SLOT(valueChanged()));
	hBox->addWidget(new QLabel("RMS Extraction Window:"));
	hBox->addWidget(m_window);
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
		QVector2D minMax = m_rep->stratiSliceAttribute()->stratiSlice()->rgtMinMax();
		m_sliceSlider->setMaximum(minMax[1]);
		m_sliceSlider->setMinimum(minMax[0]);
		m_sliceSlider->setValue(minMax[0]);

		m_sliceSpin->setMaximum(minMax[1]);
		m_sliceSpin->setMinimum(minMax[0]);
		m_sliceSpin->setValue(minMax[0]);
	}

	connect(m_rep->stratiSliceAttribute(), SIGNAL(extractionWindowChanged(unsigned int)),
			this, SLOT(extractionWindowChanged(unsigned int)));
	connect(m_rep->stratiSliceAttribute(), SIGNAL(RGTIsoValueChanged(int)), this,
			SLOT(RGTIsoValueChanged(int)));

	extractionWindowChanged(m_rep->stratiSliceAttribute()->extractionWindow());
	RGTIsoValueChanged(m_rep->stratiSliceAttribute()->currentPosition());

}
StratiSliceAmplitudePropPanel::~StratiSliceAmplitudePropPanel() {

}


void StratiSliceAmplitudePropPanel::extractionWindowChanged(unsigned int size) {
	QSignalBlocker b(m_window);
	m_window->setText(
			locale().toString(m_rep->stratiSliceAttribute()->extractionWindow()));
}
void StratiSliceAmplitudePropPanel::RGTIsoValueChanged(int pos) {
	updateSpinValue(pos, m_sliceSlider, m_sliceSpin);
}
void StratiSliceAmplitudePropPanel::updateSpinValue(int value, QSlider *slider, QSpinBox *spin) {
	QSignalBlocker b1(slider);
	QSignalBlocker b2(spin);
	slider->setValue(value);
	spin->setValue(value);
}

void StratiSliceAmplitudePropPanel::showCrossHair(int value) {
	m_rep->showCrossHair(value == Qt::Checked);
}
QWidget* StratiSliceAmplitudePropPanel::createSliceBox() {
	m_sliderBox = new QGroupBox("RGT Iso Value", this);
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

uint StratiSliceAmplitudePropPanel::getExtactionWindow() {
	bool ok;
	uint win = locale().toUInt(m_window->text(), &ok);
	if (!ok)
		return m_rep->stratiSliceAttribute()->extractionWindow();
	return win;
}

void StratiSliceAmplitudePropPanel::valueChanged() {
	uint win = getExtactionWindow();
	m_rep->stratiSliceAttribute()->setExtractionWindow(win);
}

void StratiSliceAmplitudePropPanel::sliceChanged(int value) {
	m_rep->stratiSliceAttribute()->setSlicePosition(value);
}

void StratiSliceAmplitudePropPanel::updatePalette() {
	m_palette->setPaletteHolder(m_rep->stratiSliceAttribute()->image());
}

