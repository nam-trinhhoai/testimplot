#include "fixedrgblayersfromdatasetproppanelonslice.h"

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

#include "cudaimagepaletteholder.h"
#include "fixedrgblayersfromdatasetreponslice.h"
#include "fixedrgblayersfromdataset.h"

FixedRGBLayersFromDatasetPropPanelOnSlice::FixedRGBLayersFromDatasetPropPanelOnSlice(
		FixedRGBLayersFromDatasetRepOnSlice *rep, QWidget *parent) : QWidget(parent) {
	m_rep = rep;
	QVBoxLayout *processLayout = new QVBoxLayout(this);
	//processLayout->setMargin(0);
	processLayout->setContentsMargins(0,0,0,0);

	QWidget* holder = new QWidget;
	QFormLayout* formLayout = new QFormLayout(holder);

	QLabel* label = new QLabel("Layers");
	QSlider* slider = new QSlider(Qt::Orientation::Horizontal);
	slider->setValue(m_rep->fixedRGBLayersFromDataset()->currentImageIndex());
	slider->setMinimum(0);
	slider->setMaximum(m_rep->fixedRGBLayersFromDataset()->selectedLayersKeys().size()-1);
	slider->setTickInterval(1);

	connect(slider, &QSlider::valueChanged, this, &FixedRGBLayersFromDatasetPropPanelOnSlice::changeDataKeyFromSlider);

	formLayout->addRow(label, slider);

	processLayout->addWidget(holder);

	connect(m_rep->fixedRGBLayersFromDataset()->image()->get(0), &CUDAImagePaletteHolder::dataChanged, this,
			[this, slider]() {
		QSignalBlocker b(slider);
		slider->setValue(m_rep->fixedRGBLayersFromDataset()->currentImageIndex());
	});
}

FixedRGBLayersFromDatasetPropPanelOnSlice::~FixedRGBLayersFromDatasetPropPanelOnSlice() {
}

void FixedRGBLayersFromDatasetPropPanelOnSlice::changeDataKeyFromSlider(long index) {
	m_rep->fixedRGBLayersFromDataset()->setCurrentImageIndex(index);
}

