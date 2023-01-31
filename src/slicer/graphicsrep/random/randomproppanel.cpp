#include "randomproppanel.h"

#include "randomrep.h"
#include "palettewidget.h"
#include "cudaimagepaletteholder.h"
#include "seismic3dabstractdataset.h"

#include <QHBoxLayout>
#include <QToolButton>
#include <QAction>
#include <QLabel>
#include <iostream>
#include <QCheckBox>
#include <QSlider>
#include <QSpinBox>
#include <QPushButton>

RandomPropPanel::RandomPropPanel(RandomRep *rep, QWidget *parent) :
		QWidget(parent) {
	m_rep = rep;

	QVBoxLayout *processLayout = new QVBoxLayout(this);
	//processLayout->setMargin(0);
	processLayout->setContentsMargins(0,0,0,0);
	//palette
	m_palette = new PaletteWidget(this);
	processLayout->addWidget(m_palette, 0, Qt::AlignmentFlag::AlignTop);

	if (m_rep->image()!=nullptr) {
		m_palette->setLookupTable(m_rep->image()->lookupTable());
		m_palette->setOpacity(m_rep->image()->opacity());

		//Connect the image update
		connect(m_palette, SIGNAL(rangeChanged(const QVector2D &)), m_rep->image(),
				SLOT(setRange(const QVector2D &)));
		connect(m_palette, SIGNAL(opacityChanged(float)), m_rep->image(),
				SLOT(setOpacity(float)));
		connect(m_palette, SIGNAL(lookupTableChanged(const LookupTable &)),
				m_rep->image(), SLOT(setLookupTable(const LookupTable &)));

		connect(m_rep->image(), SIGNAL(rangeChanged(const QVector2D &)), m_palette,
				SLOT(setRange(const QVector2D &)));
		connect(m_rep->image(), SIGNAL(opacityChanged(float)), m_palette,
				SLOT(setOpacity(float)));
		connect(m_rep->image(), SIGNAL(lookupTableChanged(const LookupTable &)),
				m_palette, SLOT(setLookupTable(const LookupTable &)));

		m_palette->setPaletteHolder(m_rep->image());
	}

	m_showColorScale=new QCheckBox("Show color scale",this);
	m_showColorScale->setChecked(m_rep->colorScale());
	connect(m_showColorScale, SIGNAL(stateChanged(int)), this, SLOT(showColorScale(int)));

	processLayout->addWidget(m_showColorScale, 0, Qt::AlignmentFlag::AlignTop);

	m_lockPalette = new QCheckBox("Lock Palette");
	updateLockCheckBox();
	processLayout->addWidget(m_lockPalette, 0, Qt::AlignmentFlag::AlignTop);

	connect(m_lockPalette, &QCheckBox::stateChanged, this,
			&RandomPropPanel::lockPalette);
	connect(m_palette, &PaletteWidget::rangeChanged,
				this, &RandomPropPanel::updateLockRange);

	connect(m_rep->image(),
			QOverload<const QVector2D&>::of(&CUDAImagePaletteHolder::rangeChanged), this,
			&RandomPropPanel::updateLockRange);

	Seismic3DAbstractDataset* dataset = dynamic_cast<Seismic3DAbstractDataset*>(m_rep->data());
	if (dataset!=nullptr && dataset->dimV()>1) {
		QWidget* channelHolder = new QWidget;
		QHBoxLayout* channelLayout = new QHBoxLayout;
		channelHolder->setLayout(channelLayout);

		processLayout->addWidget(channelHolder, 0, Qt::AlignmentFlag::AlignTop);

		channelLayout->addWidget(new QLabel("Channel"));

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

		connect(m_channelSlider, &QSlider::valueChanged, this, &RandomPropPanel::updateChannelInternal);
		connect(m_channelSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &RandomPropPanel::updateChannelInternal);
		connect(m_rep, &RandomRep::channelChanged, this, &RandomPropPanel::updateChannel);
	} else {
		m_channelSlider = nullptr;
		m_channelSpinBox = nullptr;
	}

	QPushButton* updateFileDynamic = new QPushButton("Write range to file");
	processLayout->addWidget(updateFileDynamic, 0, Qt::AlignmentFlag::AlignTop);

	connect(updateFileDynamic, &QPushButton::clicked, this, &RandomPropPanel::writeRangeToFile);
}
void RandomPropPanel::showColorScale(int value)
{
	m_rep->showColorScale(value==Qt::Checked);
}

void RandomPropPanel::updatePalette() {
	if (m_rep->image()!=nullptr) {
		m_palette->setPaletteHolder(m_rep->image());
	}
}

void RandomPropPanel::updateChannel(int channel) {
	if (m_channelSlider!=nullptr && m_channelSpinBox!=nullptr) {
		QSignalBlocker b1(m_channelSlider);
		m_channelSlider->setValue(channel);

		QSignalBlocker b2(m_channelSpinBox);
		m_channelSpinBox->setValue(channel);
	}
}

void RandomPropPanel::updateChannelInternal(int channel) {
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

void RandomPropPanel::updateLockCheckBox() {
	Seismic3DAbstractDataset* dataset = dynamic_cast<Seismic3DAbstractDataset*>(m_rep->data());
	bool isRangeLocked = dataset->isRangeLocked();
	int lockState = (isRangeLocked) ? Qt::Checked : Qt::Unchecked;

	QSignalBlocker b1(m_lockPalette);
	m_lockPalette->setChecked(lockState);
}

void RandomPropPanel::lockPalette(int state) {
	Seismic3DAbstractDataset* dataset = dynamic_cast<Seismic3DAbstractDataset*>(m_rep->data());
	if (state==Qt::Checked) {
		dataset->lockRange(m_rep->image()->range());
	} else {
		dataset->unlockRange();
	}
}

void RandomPropPanel::updateLockRange(const QVector2D & range) {
	Seismic3DAbstractDataset* dataset = dynamic_cast<Seismic3DAbstractDataset*>(m_rep->data());
	if (dataset->isRangeLocked()) {
		dataset->lockRange(range);
	}
}

void RandomPropPanel::writeRangeToFile() {
	Seismic3DAbstractDataset* dataset = dynamic_cast<Seismic3DAbstractDataset*>(m_rep->data());
	dataset->writeRangeToFile(m_rep->image()->range());
	dataset->lockRange(m_rep->image()->range());
}

