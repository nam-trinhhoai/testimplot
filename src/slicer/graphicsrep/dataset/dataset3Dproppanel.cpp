#include "dataset3Dproppanel.h"

#include "dataset3Dslicerep.h"
#include "palettewidget.h"
#include "cudaimagepaletteholder.h"
#include "seismic3dabstractdataset.h"

#include <QHBoxLayout>
#include <QToolButton>
#include <QAction>
#include <QLabel>
#include <iostream>
#include <QCheckBox>

#include <QHBoxLayout>
#include <QGroupBox>
#include <QSlider>
#include <QSpinBox>

Dataset3DPropPanel::Dataset3DPropPanel(Dataset3DSliceRep *rep, QWidget *parent) :
		QWidget(parent) {
	m_rep = rep;

	QVBoxLayout *processLayout = new QVBoxLayout(this);
	//processLayout->setMargin(0);
	processLayout->setContentsMargins(0,0,0,0);
	if (rep->direction() == SliceDirection::Inline)
		processLayout->addWidget(createSliceBox("Inline"), 0,
				Qt::AlignmentFlag::AlignTop);
	else
		processLayout->addWidget(createSliceBox("Xline"), 0,
				Qt::AlignmentFlag::AlignTop);

	Seismic3DAbstractDataset* dataset = dynamic_cast<Seismic3DAbstractDataset*>(m_rep->data());
	if (dataset!=nullptr && dataset->dimV()>1) {
		QGroupBox * channelHolder = new QGroupBox("DimV");
		QHBoxLayout* channelLayout = new QHBoxLayout;
		channelHolder->setLayout(channelLayout);
	//	channelLayout->setMargin(0);
		channelLayout->setContentsMargins(0,0,0,0);

		processLayout->addWidget(channelHolder, 0, Qt::AlignmentFlag::AlignTop);

		m_channelSlider = new QSlider(Qt::Horizontal);
		m_channelSlider->setMinimum(0);
		m_channelSlider->setMaximum(dataset->dimV()-1);
		m_channelSlider->setValue(m_rep->channel());

		channelLayout->addWidget(m_channelSlider);

		m_channelSpinBox = new QSpinBox;
		m_channelSpinBox->setMinimum(0);
		m_channelSpinBox->setMaximum(dataset->dimV()-1);
		m_channelSpinBox->setValue(m_rep->channel());

		channelLayout->addWidget(m_channelSpinBox);

		connect(m_channelSlider, &QSlider::valueChanged, this, &Dataset3DPropPanel::updateChannelInternal);
		connect(m_channelSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &Dataset3DPropPanel::updateChannelInternal);
		connect(m_rep, &Dataset3DSliceRep::channelChanged, this, &Dataset3DPropPanel::updateChannel);
	} else {
		m_channelSlider = nullptr;
		m_channelSpinBox = nullptr;
	}

	//palette
	m_palette = new PaletteWidget(this);
	processLayout->addWidget(m_palette, 0, Qt::AlignmentFlag::AlignTop);

	m_palette->setLookupTable(m_rep->image()->lookupTable());
	m_palette->setOpacity(m_rep->image()->opacity());

	//Connect the image update
	connect(m_palette, SIGNAL(rangeChanged(const QVector2D &)), m_rep->image(),
			SLOT(setRange(const QVector2D &)));
	connect(m_palette, SIGNAL(opacityChanged(float)), m_rep->image(),
			SLOT(setOpacity(float)));
	connect(m_palette, SIGNAL(lookupTableChanged(const LookupTable &)),
			m_rep->image(), SLOT(setLookupTable(const LookupTable &)));

	m_palette->setPaletteHolder(m_rep->image());

	QPair<QVector2D, AffineTransformation> sliceRangeAndTransfo =
			rep->sliceRangeAndTransfo();
	defineSliceMinMax(sliceRangeAndTransfo.first,
			(int) sliceRangeAndTransfo.second.a());

	connect(rep, SIGNAL(sliceWordPositionChanged(int )), this,
			SLOT(onSliceChangedRequestFromRep(int )));
}

Dataset3DPropPanel::~Dataset3DPropPanel() {

}

void Dataset3DPropPanel::onSliceChangedRequestFromRep(int val) {
	if (val == m_sliceImageSlider->value())
		return;
	defineSliceVal(val);
}

QWidget* Dataset3DPropPanel::createSliceBox(const QString &title) {
	QGroupBox *sliderBox = new QGroupBox(title, this);

	m_sliceImageSlider = new QSlider(Qt::Orientation::Horizontal, this);
	m_sliceImageSlider->setSingleStep(1);
	m_sliceImageSlider->setTickInterval(10);
	m_sliceImageSlider->setMinimum(0);
	m_sliceImageSlider->setMaximum(1);
	m_sliceImageSlider->setValue(0);

	m_sliceSpin = new QSpinBox();
	m_sliceSpin->setMinimum(0);
	m_sliceSpin->setMaximum(1);
	m_sliceSpin->setSingleStep(1);
	m_sliceSpin->setValue(0);

	m_sliceSpin->setWrapping(false);

	connect(m_sliceSpin, SIGNAL(valueChanged(int)), this,
			SLOT(sliceChanged(int )));
	connect(m_sliceImageSlider, SIGNAL(valueChanged(int)), this,
			SLOT(sliceChanged(int)));

	QHBoxLayout *hBox = new QHBoxLayout(sliderBox);
	//hBox->setMargin(0);
	hBox->setContentsMargins(0,0,0,0);
	hBox->addWidget(m_sliceImageSlider);
	hBox->addWidget(m_sliceSpin);
	return sliderBox;
}

void Dataset3DPropPanel::defineSliceMinMax(const QVector2D &imageMinMax,
		int step) {
	QSignalBlocker b1(m_sliceImageSlider);
	m_sliceImageSlider->setMinimum((int) imageMinMax.x());
	m_sliceImageSlider->setMaximum((int) imageMinMax.y());
	m_sliceImageSlider->setSingleStep(step);
	m_sliceImageSlider->setTickInterval(step);
	m_sliceImageSlider->setPageStep(step);

	QSignalBlocker b2(m_sliceSpin);
	m_sliceSpin->setMinimum((int) imageMinMax.x());
	m_sliceSpin->setMaximum((int) imageMinMax.y());
	m_sliceSpin->setSingleStep(step);
}

void Dataset3DPropPanel::defineSliceVal(int image) {
	QSignalBlocker b1(m_sliceImageSlider);
	m_sliceImageSlider->setValue(image);

	QSignalBlocker b2(m_sliceSpin);
	m_sliceSpin->setValue(image);
}

void Dataset3DPropPanel::sliceChanged(int val) {
	int realVal = val;
	int reste = val % (int) m_rep->sliceRangeAndTransfo().second.a();
	if (reste != 0) {
		realVal = val + reste;
	}

	m_rep->setSliceWorldPosition(realVal);
	defineSliceVal(realVal);
}

void Dataset3DPropPanel::updatePalette() {
	m_palette->setPaletteHolder(m_rep->image());
}

void Dataset3DPropPanel::updateChannel(int channel) {
	if (m_channelSlider!=nullptr && m_channelSpinBox!=nullptr) {
		QSignalBlocker b1(m_channelSlider);
		m_channelSlider->setValue(channel);

		QSignalBlocker b2(m_channelSpinBox);
		m_channelSpinBox->setValue(channel);
	}
}

void Dataset3DPropPanel::updateChannelInternal(int channel) {
	if (channel!=m_rep->channel()) {
		if (m_channelSlider!=nullptr && m_channelSpinBox!=nullptr) {
			QSignalBlocker b1(m_channelSlider);
			m_channelSlider->setValue(channel);

			QSignalBlocker b2(m_channelSpinBox);
			m_channelSpinBox->setValue(channel);
		}

		m_rep->setChannel(channel);
	}
}

