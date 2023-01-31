#include "computationoperatordatasetproppanel.h"

#include "computationoperatordatasetrep.h"
#include "palettewidget.h"
#include "cudaimagepaletteholder.h"
#include "computationoperatordataset.h"

#include <QHBoxLayout>
#include <QToolButton>
#include <QAction>
#include <QLabel>
#include <iostream>
#include <QCheckBox>
#include <QSpinBox>
#include <QSlider>
#include <QPushButton>
#include <QMessageBox>

ComputationOperatorDatasetPropPanel::ComputationOperatorDatasetPropPanel(ComputationOperatorDatasetRep *rep, QWidget *parent) :
		QWidget(parent) {
	m_rep = rep;

	QVBoxLayout *processLayout = new QVBoxLayout(this);
	//processLayout->setMargin(0);
	processLayout->setContentsMargins(0,0,0,0);

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

	connect(m_rep->image(), SIGNAL(rangeChanged(const QVector2D &)), m_palette,
			SLOT(setRange(const QVector2D &)));
	connect(m_rep->image(), SIGNAL(opacityChanged(float)), m_palette,
			SLOT(setOpacity(float)));
	connect(m_rep->image(), SIGNAL(lookupTableChanged(const LookupTable &)),
			m_palette, SLOT(setLookupTable(const LookupTable &)));

	m_palette->setPaletteHolder(m_rep->image());

	m_showColorScale=new QCheckBox("Show color scale",this);
	m_showColorScale->setChecked(m_rep->colorScale());
	connect(m_showColorScale, SIGNAL(stateChanged(int)), this, SLOT(showColorScale(int)));

	processLayout->addWidget(m_showColorScale, 0, Qt::AlignmentFlag::AlignTop);

	m_lockPalette = new QCheckBox("Lock Palette");
	updateLockCheckBox();
	processLayout->addWidget(m_lockPalette, 0, Qt::AlignmentFlag::AlignTop);

	connect(m_lockPalette, &QCheckBox::stateChanged, this,
			&ComputationOperatorDatasetPropPanel::lockPalette);
	connect(m_palette, &PaletteWidget::rangeChanged,
				this, &ComputationOperatorDatasetPropPanel::updateLockRange);

	connect(m_rep->image(),
			QOverload<const QVector2D&>::of(&CUDAImagePaletteHolder::rangeChanged), this,
			&ComputationOperatorDatasetPropPanel::updateLockRange);

	ComputationOperatorDataset* dataset = dynamic_cast<ComputationOperatorDataset*>(m_rep->data());
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

		connect(m_channelSlider, &QSlider::valueChanged, this, &ComputationOperatorDatasetPropPanel::updateChannelInternal);
		connect(m_channelSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &ComputationOperatorDatasetPropPanel::updateChannelInternal);
		connect(m_rep, &ComputationOperatorDatasetRep::channelChanged, this, &ComputationOperatorDatasetPropPanel::updateChannel);
	} else {
		m_channelSlider = nullptr;
		m_channelSpinBox = nullptr;
	}
}
void ComputationOperatorDatasetPropPanel::showColorScale(int value)
{
	m_rep->showColorScale(value==Qt::Checked);
}
void ComputationOperatorDatasetPropPanel::updatePalette() {
	m_palette->setPaletteHolder(m_rep->image());
}

void ComputationOperatorDatasetPropPanel::updateChannel(int channel) {
	if (m_channelSlider!=nullptr && m_channelSpinBox!=nullptr) {
		QSignalBlocker b1(m_channelSlider);
		m_channelSlider->setValue(channel);

		QSignalBlocker b2(m_channelSpinBox);
		m_channelSpinBox->setValue(channel);
	}
}

void ComputationOperatorDatasetPropPanel::updateChannelInternal(int channel) {
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

void ComputationOperatorDatasetPropPanel::updateLockCheckBox() {
	ComputationOperatorDataset* dataset = dynamic_cast<ComputationOperatorDataset*>(m_rep->data());
	bool isRangeLocked = dataset->isRangeLocked();
	int lockState = (isRangeLocked) ? Qt::Checked : Qt::Unchecked;

	QSignalBlocker b1(m_lockPalette);
	m_lockPalette->setChecked(lockState);
}

void ComputationOperatorDatasetPropPanel::lockPalette(int state) {
	ComputationOperatorDataset* dataset = dynamic_cast<ComputationOperatorDataset*>(m_rep->data());
	if (state==Qt::Checked) {
		dataset->lockRange(m_rep->image()->range());
	} else {
		dataset->unlockRange();
	}
}

void ComputationOperatorDatasetPropPanel::updateLockRange(const QVector2D & range) {
	ComputationOperatorDataset* dataset = dynamic_cast<ComputationOperatorDataset*>(m_rep->data());
	if (dataset->isRangeLocked()) {
		dataset->lockRange(range);
	}
}

