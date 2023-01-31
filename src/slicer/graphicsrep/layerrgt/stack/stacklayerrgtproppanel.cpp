#include "stacklayerrgtproppanel.h"

#include <iostream>
#include "stacklayerrgtrep.h"
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

StackLayerRGTPropPanel::StackLayerRGTPropPanel(StackLayerRGTRep *rep, bool for3D, QWidget *parent) :
		QWidget(parent) {
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


	m_showCrossHair = new QCheckBox("Show crosshair", this);
	m_showCrossHair->setChecked(m_rep->crossHair());
	connect(m_showCrossHair, SIGNAL(stateChanged(int)), this,
			SLOT(showCrossHair(int)));

	processLayout->addWidget(m_showCrossHair, 0,
			Qt::AlignmentFlag::AlignTop);

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

	connect(m_rep->image(),
			SIGNAL(rangeChanged(const QVector2D &)), this,
			SLOT(updateLockRange(const QVector2D &)));
	connect(m_rep->image(),
			SIGNAL(lookupTableChanged(const LookupTable &)), this,
			SLOT(updateLockLookupTable(const LookupTable &)));

	connect(m_rep, &StackLayerRGTRep::sliceIJPositionChanged, this,
			&StackLayerRGTPropPanel::updateLockCheckBox);
}

StackLayerRGTPropPanel::~StackLayerRGTPropPanel() {

}

void StackLayerRGTPropPanel::pick() {
	if (m_pickButton->isChecked()) {
		m_pickingTask = new PointPickingTask(this);
		m_rep->view()->registerPickingTask(m_pickingTask);
	} else {
		m_rep->view()->unregisterPickingTask(m_pickingTask);
		delete m_pickingTask;
		m_pickingTask = nullptr;
	}
}

void StackLayerRGTPropPanel::pointPicked(double worldX, double worldY,
		Qt::MouseButton button, Qt::KeyboardModifiers keys) {
	std::cout << "Picked position:" << worldX << "\t" << worldY << std::endl;

}

void StackLayerRGTPropPanel::showCrossHair(int value) {
	m_rep->showCrossHair(value == Qt::Checked);
}

void StackLayerRGTPropPanel::updatePalette() {
	CUDAImagePaletteHolder* image = m_rep->image();
	m_palette->setPaletteHolder(image);
}

void StackLayerRGTPropPanel::updateLockCheckBox() {
	QString label = m_rep->getCurrentLabel();
	const std::pair<bool, LayerSlice::PaletteParameters>& lockedPalette =
			m_rep->layerSlice()->getLockedPalette(label);
	int lockState = (lockedPalette.first) ? Qt::Checked : Qt::Unchecked;

	QSignalBlocker b1(m_lockPalette);
	m_lockPalette->setChecked(lockState);
}

void StackLayerRGTPropPanel::lockPalette(int state) {
	if (state==Qt::Checked) {
		LayerSlice::PaletteParameters params;
		params.range = m_rep->image()->range();
		params.lookupTable = m_rep->image()->lookupTable();
		m_rep->layerSlice()->lockPalette(m_rep->getCurrentLabel(), params);
	} else {
		m_rep->layerSlice()->unlockPalette(m_rep->getCurrentLabel());
	}
}

void StackLayerRGTPropPanel::updateLockRange(const QVector2D & range) {
	QString label = m_rep->getCurrentLabel();
	std::pair<bool, LayerSlice::PaletteParameters> lockedPalette =
		m_rep->layerSlice()->getLockedPalette(label);
	if (lockedPalette.first) {
		LayerSlice::PaletteParameters params = lockedPalette.second;
		params.range = range;
		m_rep->layerSlice()->lockPalette(label, params);
	}
}

void StackLayerRGTPropPanel::updateLockLookupTable(const LookupTable& lookupTable) {
	QString label = m_rep->getCurrentLabel();
	std::pair<bool, LayerSlice::PaletteParameters> lockedPalette =
		m_rep->layerSlice()->getLockedPalette(label);
	if (lockedPalette.first) {
		LayerSlice::PaletteParameters params = lockedPalette.second;
		params.lookupTable = lookupTable;
		m_rep->layerSlice()->lockPalette(label, params);
	}
}

