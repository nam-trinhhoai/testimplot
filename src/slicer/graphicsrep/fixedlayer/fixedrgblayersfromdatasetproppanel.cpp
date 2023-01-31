#include "fixedrgblayersfromdatasetproppanel.h"

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
#include "fixedrgblayersfromdatasetrep.h"
#include "fixedrgblayersfromdataset.h"

FixedRGBLayersFromDatasetPropPanel::FixedRGBLayersFromDatasetPropPanel(FixedRGBLayersFromDatasetRep *rep, QWidget *parent) :
		QWidget(parent) {
	m_rep = rep;
	QVBoxLayout *processLayout = new QVBoxLayout(this);
	//processLayout->setMargin(0);
	processLayout->setContentsMargins(0,0,0,0);

	//Palettes
	m_palette = new RGBPaletteWidget(this);
	processLayout->addWidget(m_palette, 0, Qt::AlignmentFlag::AlignTop);
	m_palette->setPaletteHolders(m_rep->fixedRGBLayersFromDataset()->image()->holders());
	m_palette->setOpacity(m_rep->fixedRGBLayersFromDataset()->image()->opacity());

	//Connect the image update
	connect(m_palette, SIGNAL(rangeChanged(unsigned int ,const QVector2D & )),
			m_rep->fixedRGBLayersFromDataset()->image(),
			SLOT(setRange(unsigned int ,const QVector2D & )));
	connect(m_palette, SIGNAL(opacityChanged(float)),
			m_rep->fixedRGBLayersFromDataset()->image(), SLOT(setOpacity(float)));

	connect(m_rep->fixedRGBLayersFromDataset()->image(),
			SIGNAL(rangeChanged(unsigned int ,const QVector2D & )), m_palette,
			SLOT(setRange(unsigned int ,const QVector2D & )));
	connect(m_rep->fixedRGBLayersFromDataset()->image(), SIGNAL(opacityChanged(float)),
			m_palette, SLOT(setOpacity(float)));

	//Colors
//	processLayout->addWidget(createImageChooserWidget());
	QWidget* holder = new QWidget;
	QFormLayout* formLayout = new QFormLayout(holder);

	QLabel* label = new QLabel("Layers");
	QSlider* slider = new QSlider(Qt::Orientation::Horizontal);
	slider->setValue(m_rep->fixedRGBLayersFromDataset()->currentImageIndex());
	slider->setMinimum(0);
	slider->setMaximum(m_rep->fixedRGBLayersFromDataset()->selectedLayersKeys().size()-1);
	slider->setTickInterval(1);

	connect(slider, &QSlider::valueChanged, this, &FixedRGBLayersFromDatasetPropPanel::changeDataKeyFromSlider);

	formLayout->addRow(label, slider);

	processLayout->addWidget(holder);

	connect(m_rep->fixedRGBLayersFromDataset()->image()->get(0), &CUDAImagePaletteHolder::dataChanged, this,
			[this, slider]() {
		QSignalBlocker b(slider);
		slider->setValue(m_rep->fixedRGBLayersFromDataset()->currentImageIndex());
	});

//	updateSpectrum();

	//Listen to data modification
//	connect(m_rep->rgbLayerSlice(), SIGNAL(frequencyChanged()), this,
//			SLOT(frequencyChanged()));
//	connect(m_rep->rgbLayerSlice()->layerSlice(), &LayerSlice::computationFinished, [this]() {
//		if (m_rep->rgbLayerSlice()->redIndex()!=m_redSpin->value()) {
//			QSignalBlocker b1(m_redSpin);
//			QSignalBlocker b2(m_redSlider);
//			m_redSlider->setValue(m_rep->rgbLayerSlice()->redIndex());
//
//			m_redSpin->setValue(m_rep->rgbLayerSlice()->redIndex());
//		}
//		if (m_rep->rgbLayerSlice()->layerSlice()->getNbOutputSlices()!=m_redSpin->maximum()+1) {
//			QSignalBlocker b1(m_redSpin);
//			QSignalBlocker b2(m_redSlider);
//			m_redSlider->setMaximum(m_rep->rgbLayerSlice()->layerSlice()->getNbOutputSlices()-1);
//
//			m_redSpin->setMaximum(m_rep->rgbLayerSlice()->layerSlice()->getNbOutputSlices()-1);
//		}
//		if (m_rep->rgbLayerSlice()->blueIndex()!=m_blueSpin->value()) {
//			QSignalBlocker b1(m_blueSpin);
//			QSignalBlocker b2(m_blueSlider);
//			m_blueSlider->setValue(m_rep->rgbLayerSlice()->blueIndex());
//
//			m_blueSpin->setValue(m_rep->rgbLayerSlice()->blueIndex());
//		}
//		if (m_rep->rgbLayerSlice()->layerSlice()->getNbOutputSlices()!=m_blueSpin->maximum()+1) {
//			QSignalBlocker b1(m_blueSpin);
//			QSignalBlocker b2(m_blueSlider);
//			m_blueSlider->setMaximum(m_rep->rgbLayerSlice()->layerSlice()->getNbOutputSlices()-1);
//
//			m_blueSpin->setMaximum(m_rep->rgbLayerSlice()->layerSlice()->getNbOutputSlices()-1);
//		}
//		if (m_rep->rgbLayerSlice()->redIndex()!=m_greenSpin->value()) {
//			QSignalBlocker b1(m_greenSpin);
//			QSignalBlocker b2(m_greenSlider);
//			m_greenSlider->setValue(m_rep->rgbLayerSlice()->greenIndex());
//
//			m_greenSpin->setValue(m_rep->rgbLayerSlice()->greenIndex());
//		}
//		if (m_rep->rgbLayerSlice()->layerSlice()->getNbOutputSlices()!=m_greenSpin->maximum()+1) {
//			QSignalBlocker b1(m_greenSpin);
//			QSignalBlocker b2(m_greenSlider);
//			m_greenSlider->setMaximum(m_rep->rgbLayerSlice()->layerSlice()->getNbOutputSlices()-1);
//
//			m_greenSpin->setMaximum(m_rep->rgbLayerSlice()->layerSlice()->getNbOutputSlices()-1);
//		}
//	});
}

FixedRGBLayersFromDatasetPropPanel::~FixedRGBLayersFromDatasetPropPanel() {
}

void FixedRGBLayersFromDatasetPropPanel::changeDataKeyFromSlider(long index) {
	m_rep->fixedRGBLayersFromDataset()->setCurrentImageIndex(index);
}


QWidget* FixedRGBLayersFromDatasetPropPanel::createImageChooserWidget() {
//	//Frequency chooser
//	QWidget *colorWidget = new QWidget(this);
//	QFormLayout *colorLayout = new QFormLayout(colorWidget);
//
//	QLabel *r = new QLabel("Red");
//	r->setPixmap(QIcon(":/palette/icons/red.png").pixmap(QSize(16, 16)));
//	QWidget *redWidget = new QWidget(this);
//
//
	return nullptr; //colorWidget;
}

void FixedRGBLayersFromDatasetPropPanel::updatePalette(int i) {
	m_palette->setPaletteHolder(i, m_rep->fixedRGBLayersFromDataset()->image()->get(i));
}

