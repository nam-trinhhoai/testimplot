#include "stratislicergbproppanel.h"

#include <QDebug>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QSlider>
#include <QSpinBox>
#include <QSpacerItem>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <QHBoxLayout>
#include <QMouseEvent>
#include <QLineEdit>
#include <QLabel>
#include <QFormLayout>
#include <QComboBox>
#include <QStringListModel>
#include <QListView>

#include <iostream>
#include <sstream>

#include "rgbpalettewidget.h"
#include "cudaimagepaletteholder.h"
#include "cudargbimage.h"
#include "stratislicergbattributerep.h"
#include "rgbstratisliceattribute.h"
#include "stratislice.h"

StratiSliceRGBPropPanel::StratiSliceRGBPropPanel(StratiSliceRGBAttributeRep *rep, QWidget *parent) :
		QWidget(parent) {
	m_rep = rep;
	QVBoxLayout *processLayout = new QVBoxLayout(this);
	//processLayout->setMargin(0);
	processLayout->setContentsMargins(0,0,0,0);

	//Palettes
	m_palette = new RGBPaletteWidget(this);
	processLayout->addWidget(m_palette, 0, Qt::AlignmentFlag::AlignTop);
	m_palette->setPaletteHolders(m_rep->stratiSliceAttribute()->image()->holders());
	m_palette->setOpacity(m_rep->stratiSliceAttribute()->image()->opacity());

	//Connect the image update
	connect(m_palette, SIGNAL(rangeChanged(unsigned int ,const QVector2D & )),
			m_rep->stratiSliceAttribute()->image(),
			SLOT(setRange(unsigned int ,const QVector2D & )));
	connect(m_palette, SIGNAL(opacityChanged(float)),
			m_rep->stratiSliceAttribute()->image(), SLOT(setOpacity(float)));

	connect(m_rep->stratiSliceAttribute()->image(),
			SIGNAL(rangeChanged(unsigned int ,const QVector2D & )), m_palette,
			SLOT(setRange(unsigned int ,const QVector2D & )));
	connect(m_rep->stratiSliceAttribute()->image(), SIGNAL(opacityChanged(float)),
			m_palette, SLOT(setOpacity(float)));

	//Slice position
	m_sliceSlider = new QSlider(Qt::Orientation::Horizontal, this);
	//m_sliceSlider->setTracking(false);
	m_sliceSpin = new QSpinBox();

	QWidget *sliceBox = createSlideSpinBox("RGT Iso Value", m_sliceSlider,
			m_sliceSpin);
	connect(m_sliceSlider, SIGNAL(valueChanged(int)), this,
			SLOT(sliceChanged(int)));
	connect(m_sliceSpin, SIGNAL(valueChanged(int)), this,
			SLOT(sliceChanged(int)));
	sliceBox->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);

	processLayout->addWidget(sliceBox, 0, Qt::AlignmentFlag::AlignTop);

	//Window
	processLayout->addWidget(createWindowParameterWidget(), 0,
			Qt::AlignmentFlag::AlignTop);

	//Colors
	processLayout->addWidget(createFreqChooserWidget());

	{
		QSignalBlocker b1(m_sliceSpin);
		QSignalBlocker b2(m_sliceSlider);
		QVector2D minMax = m_rep->stratiSliceAttribute()->stratiSlice()->rgtMinMax();
		updateSliderSpin(minMax[0], minMax[1], m_sliceSlider, m_sliceSpin);
	}

	//Listen to data modification
	connect(m_rep->stratiSliceAttribute(), SIGNAL(extractionWindowChanged(unsigned int)),
			this, SLOT(extractionWindowChanged(unsigned int)));
	connect(m_rep->stratiSliceAttribute(), SIGNAL(RGTIsoValueChanged(int)), this,
			SLOT(RGTIsoValueChanged(int)));
	connect(m_rep->stratiSliceAttribute(), SIGNAL(frequencyChanged()), this,
			SLOT(frequencyChanged()));

	//initialize
	extractionWindowChanged(m_rep->stratiSliceAttribute()->extractionWindow());
	RGTIsoValueChanged(m_rep->stratiSliceAttribute()->currentPosition());
}

StratiSliceRGBPropPanel::~StratiSliceRGBPropPanel() {
}

void StratiSliceRGBPropPanel::extractionWindowChanged(unsigned int size) {
	updateSpectrum(size);
	QSignalBlocker b(m_window);
	m_window->setText(
			locale().toString(m_rep->stratiSliceAttribute()->extractionWindow()));
}
void StratiSliceRGBPropPanel::RGTIsoValueChanged(int pos) {
	updateSpinValue(pos, m_sliceSlider, m_sliceSpin);
}
void StratiSliceRGBPropPanel::frequencyChanged() {
	updateSpinValue(m_rep->stratiSliceAttribute()->redIndex(), m_redSlider, m_redSpin);
	updateSpinValue(m_rep->stratiSliceAttribute()->greenIndex(), m_greenSlider,
			m_greenSpin);
	updateSpinValue(m_rep->stratiSliceAttribute()->blueIndex(), m_blueSlider,
			m_blueSpin);
}

QWidget* StratiSliceRGBPropPanel::createSlideSpinBox(QString title, QSlider *slider,
		QSpinBox *spin) {
	QGroupBox *sliderBox = new QGroupBox(title, this);
	sliderBox->setContentsMargins(0, 0, 0, 0);

	createlinkedSliderSpin(sliderBox, slider, spin);

	return sliderBox;
}

QWidget* StratiSliceRGBPropPanel::createWindowParameterWidget() {
	QWidget *rangeWidget = new QWidget(this);
	QHBoxLayout *hBox = new QHBoxLayout(rangeWidget);
	//hBox->setMargin(0);
	hBox->setContentsMargins(0,0,0,0);
	m_window = new QLineEdit(this);
	m_window->setLocale(QLocale::C);
	m_window->setText(
			locale().toString(m_rep->stratiSliceAttribute()->extractionWindow()));

	connect(m_window, SIGNAL(returnPressed()), this, SLOT(valueChanged()));
	hBox->addWidget(new QLabel("SpecDecomp Extraction Window:"));
	hBox->addWidget(m_window);
	return rangeWidget;
}

QWidget* StratiSliceRGBPropPanel::createFreqChooserWidget() {
	//Frequency chooser
	QWidget *colorWidget = new QWidget(this);
	QFormLayout *colorLayout = new QFormLayout(colorWidget);

	QLabel *r = new QLabel("Red");
	r->setPixmap(QIcon(":/palette/icons/red.png").pixmap(QSize(16, 16)));
	QWidget *redWidget = new QWidget(this);
	m_redSlider = new QSlider(Qt::Orientation::Horizontal, this);
	m_redSpin = new QSpinBox();
	createlinkedSliderSpin(redWidget, m_redSlider, m_redSpin);

	colorLayout->addRow(r, redWidget);

	QLabel *g = new QLabel("Green");
	g->setPixmap(QIcon(":/palette/icons/green.png").pixmap(QSize(16, 16)));
	QWidget *greenWidget = new QWidget(this);
	m_greenSlider = new QSlider(Qt::Orientation::Horizontal, this);
	m_greenSpin = new QSpinBox();
	createlinkedSliderSpin(greenWidget, m_greenSlider, m_greenSpin);
	colorLayout->addRow(g, greenWidget);

	QLabel *b = new QLabel("Green");
	b->setPixmap(QIcon(":/palette/icons/blue.png").pixmap(QSize(16, 16)));
	QWidget *blueWidget = new QWidget(this);
	m_blueSlider = new QSlider(Qt::Orientation::Horizontal, this);
	m_blueSpin = new QSpinBox();
	createlinkedSliderSpin(blueWidget, m_blueSlider, m_blueSpin);
	colorLayout->addRow(b, blueWidget);

	connect(m_redSlider, SIGNAL(valueChanged(int)), this,
			SLOT(setRedIndex(int)));
	connect(m_redSpin, SIGNAL(valueChanged(int)), this, SLOT(setRedIndex(int)));

	connect(m_greenSlider, SIGNAL(valueChanged(int)), this,
			SLOT(setGreenIndex(int)));
	connect(m_greenSpin, SIGNAL(valueChanged(int)), this,
			SLOT(setGreenIndex(int)));

	connect(m_blueSlider, SIGNAL(valueChanged(int)), this,
			SLOT(setBlueIndex(int)));
	connect(m_blueSpin, SIGNAL(valueChanged(int)), this,
			SLOT(setBlueIndex(int)));

	return colorWidget;
}

void StratiSliceRGBPropPanel::updatePalette(int i) {
	m_palette->setPaletteHolder(i, m_rep->stratiSliceAttribute()->image()->get(i));
}

void StratiSliceRGBPropPanel::updateSpinValue(int value, QSlider *slider, QSpinBox *spin) {
	QSignalBlocker b1(slider);
	QSignalBlocker b2(spin);
	slider->setValue(value);
	spin->setValue(value);
}

void StratiSliceRGBPropPanel::setRedIndex(int value) {
	m_rep->stratiSliceAttribute()->setRedIndex(value);
}
void StratiSliceRGBPropPanel::setGreenIndex(int value) {
	m_rep->stratiSliceAttribute()->setGreenIndex(value);
}
void StratiSliceRGBPropPanel::setBlueIndex(int value) {
	m_rep->stratiSliceAttribute()->setBlueIndex(value);
}

void StratiSliceRGBPropPanel::createlinkedSliderSpin(QWidget *parent, QSlider *slider,
		QSpinBox *spin) {
	slider->setSingleStep(1);
	slider->setTickInterval(10);
	slider->setMinimum(0);
	slider->setMaximum(1);
	slider->setValue(0);

	spin->setMinimum(0);
	spin->setMaximum(10000);
	spin->setSingleStep(1);
	spin->setValue(0);
	spin->setWrapping(false);

	QHBoxLayout *hBox = new QHBoxLayout(parent);
	//hBox->setMargin(0);
	hBox->setContentsMargins(0,0,0,0);
	hBox->addWidget(slider);
	hBox->addWidget(spin);
}

uint StratiSliceRGBPropPanel::getExtactionWindow() {
	bool ok;
	uint win = locale().toUInt(m_window->text(), &ok);
	if (!ok)
		return m_rep->stratiSliceAttribute()->extractionWindow();
	return win;
}

void StratiSliceRGBPropPanel::sliceChanged(int value) {
	m_rep->stratiSliceAttribute()->setSlicePosition(value);
}

void StratiSliceRGBPropPanel::setSlicePosition(int pos) {
	QSignalBlocker b1(m_sliceSlider);
	int realPos = pos;
	if (pos < m_sliceSlider->minimum())
		realPos = m_sliceSlider->minimum();
	if (pos > m_sliceSlider->maximum())
		realPos = m_sliceSlider->maximum();
	m_sliceSlider->setValue(realPos);
}

void StratiSliceRGBPropPanel::valueChanged() {
	uint win = getExtactionWindow();
	m_rep->stratiSliceAttribute()->setExtractionWindow(win);
}

void StratiSliceRGBPropPanel::updateSliderSpin(int min, int max, QSlider *slider,
		QSpinBox *spin) {
	slider->setMaximum(max);
	slider->setMinimum(min);
	slider->setValue(min);

	spin->setMaximum(max);
	spin->setMinimum(min);
	spin->setValue(min);
}

void StratiSliceRGBPropPanel::updateSpectrum(unsigned int w) {
	QSignalBlocker redBlocker(m_redSlider);
	QSignalBlocker red1Blocker(m_redSpin);
	QSignalBlocker greenBlocker(m_greenSlider);
	QSignalBlocker green1Blocker(m_greenSpin);
	QSignalBlocker blueBlocker(m_blueSlider);
	QSignalBlocker blue1Blocker(m_blueSpin);

	//Frequency chooser
	int freqCount = (w / 2 + 1) -1;

	QStringList fList;
	for (int i = 0; i < freqCount; i++)
		fList << "f" + locale().toString(i);

	updateSliderSpin(0, freqCount, m_redSlider, m_redSpin);
	updateSliderSpin(0, freqCount, m_greenSlider, m_greenSpin);
	updateSliderSpin(0, freqCount, m_blueSlider, m_blueSpin);

	int redIndex = m_rep->stratiSliceAttribute()->redIndex();
	m_redSlider->setValue(redIndex);
	m_redSpin->setValue(redIndex);

	int greenIndex = m_rep->stratiSliceAttribute()->greenIndex();
	m_greenSlider->setValue(greenIndex);
	m_greenSpin->setValue(greenIndex);

	int blueIndex = m_rep->stratiSliceAttribute()->blueIndex();
	m_blueSlider->setValue(blueIndex);
	m_blueSpin->setValue(blueIndex);
}

