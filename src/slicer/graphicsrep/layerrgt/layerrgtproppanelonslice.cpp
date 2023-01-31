#include "layerrgtproppanelonslice.h"

#include <iostream>
#include "layerrgtreponslice.h"
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
#include <QMessageBox>
#include "abstractinnerview.h"
#include "pointpickingtask.h"

LayerRGTPropPanelOnSlice::LayerRGTPropPanelOnSlice(LayerRGTRepOnSlice *rep,
		QWidget *parent) :
		QWidget(parent) {
	m_rep = rep;
	m_pickingTask = nullptr;
	QVBoxLayout *processLayout = new QVBoxLayout(this);
	//processLayout->setMargin(0);
	processLayout->setContentsMargins(0,0,0,0);
	processLayout->addWidget(createSliceBox(), 0, Qt::AlignmentFlag::AlignTop);

	m_pickButton = new QToolButton(this);
	QAction *pickAction = new QAction(tr("Pick"), this);
	pickAction->setCheckable(true);
	m_pickButton->setDefaultAction(pickAction);
	processLayout->addWidget(m_pickButton, 0, Qt::AlignmentFlag::AlignTop);
	connect(pickAction, SIGNAL(triggered()), this, SLOT(pick()));

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

	{
		QSignalBlocker b1(m_sliceSpin);
		QSignalBlocker b2(m_sliceSlider);
		QVector2D minMax = m_rep->layerSlice()->rgtMinMax();
		m_sliceSlider->setMaximum(minMax[1]);
		m_sliceSlider->setMinimum(minMax[0]);
		m_sliceSlider->setValue(minMax[0]);

		m_sliceSpin->setMaximum(minMax[1]);
		m_sliceSpin->setMinimum(minMax[0]);
		m_sliceSpin->setValue(minMax[0]);
	}

	connect(m_rep->layerSlice(), SIGNAL(extractionWindowChanged(unsigned int)),
			this, SLOT(extractionWindowChanged(unsigned int)));
	connect(m_rep->layerSlice(), SIGNAL(RGTIsoValueChanged(int)), this,
			SLOT(RGTIsoValueChanged(int)));

	extractionWindowChanged(m_rep->layerSlice()->extractionWindow());
	RGTIsoValueChanged(m_rep->layerSlice()->currentPosition());
}
LayerRGTPropPanelOnSlice::~LayerRGTPropPanelOnSlice() {

}

void LayerRGTPropPanelOnSlice::unregisterPickingTask() {
	if (m_pickingTask != nullptr) {
		m_rep->view()->unregisterPickingTask(m_pickingTask);
		disconnect(m_pickingTask,
				SIGNAL(
						pointPicked(double,double ,Qt::MouseButton ,Qt::KeyboardModifiers,const QVector<PickingInfo> & )),
				this,
				SLOT(
						pointPicked(double,double ,Qt::MouseButton ,Qt::KeyboardModifiers,const QVector<PickingInfo> & )));
		delete m_pickingTask;
		m_pickingTask = nullptr;
	}
}

void LayerRGTPropPanelOnSlice::pick() {
	if (m_pickButton->isChecked()) {
		//Register on task only
		if (m_pickingTask != nullptr)
			return;
		m_pickingTask = new PointPickingTask(this);
		connect(m_pickingTask,
				SIGNAL(
						pointPicked(double,double ,Qt::MouseButton ,Qt::KeyboardModifiers,const QVector<PickingInfo> & )),
				this,
				SLOT(
						pointPicked(double,double ,Qt::MouseButton ,Qt::KeyboardModifiers,const QVector<PickingInfo> &)));
		m_rep->view()->registerPickingTask(m_pickingTask);
	} else {
		unregisterPickingTask();
	}
}

void LayerRGTPropPanelOnSlice::pointPicked(double worldX, double worldY,
		Qt::MouseButton button, Qt::KeyboardModifiers keys,
		const QVector<PickingInfo> &info) {
	if (button != Qt::MouseButton::LeftButton)
		return;
	bool found = false;
	double rgtValue;
	for (PickingInfo i : info) {
		if (i.uuid() == m_rep->layerSlice()->rgtID() && !i.value().empty()) {
			rgtValue = i.value()[0];
			found = true;
			break;
		}
	}
	if (!found) {
		QMessageBox *msgBox = new QMessageBox(this);
		msgBox->setIcon(QMessageBox::Information);
		msgBox->setText("Associated RGT Slice must be diplayed to pick on it");
		msgBox->setWindowTitle("Picking Info");
		msgBox->setAttribute(Qt::WA_DeleteOnClose);
		msgBox->setModal(false);
		msgBox->show();
		return;
	}

	//update RGT
	m_rep->layerSlice()->setSlicePosition((int) rgtValue);
}

void LayerRGTPropPanelOnSlice::extractionWindowChanged(unsigned int size) {
//	QSignalBlocker b(m_window);
//	m_window->setText(
//			locale().toString(m_rep->layerSlice()->extractionWindow()));
}
void LayerRGTPropPanelOnSlice::RGTIsoValueChanged(int pos) {
	updateSpinValue(pos, m_sliceSlider, m_sliceSpin);
}
void LayerRGTPropPanelOnSlice::updateSpinValue(int value, QSlider *slider,
		QSpinBox *spin) {
	QSignalBlocker b1(slider);
	QSignalBlocker b2(spin);
	slider->setValue(value);
	spin->setValue(value);
}

QWidget* LayerRGTPropPanelOnSlice::createSliceBox() {
	QGroupBox * sliderBox = new QGroupBox("RGT Iso Value", this);
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

	QHBoxLayout *hBox = new QHBoxLayout(sliderBox);
	//hBox->setMargin(0);
	hBox->setContentsMargins(0,0,0,0);
	hBox->addWidget(m_sliceSlider);
	hBox->addWidget(m_sliceSpin);
	return sliderBox;
}

uint LayerRGTPropPanelOnSlice::getExtactionWindow() {
	bool ok;
	uint win = locale().toUInt(m_window->text(), &ok);
	if (!ok)
		return m_rep->layerSlice()->extractionWindow();
	return win;
}

void LayerRGTPropPanelOnSlice::valueChanged() {
	uint win = getExtactionWindow();
	m_rep->layerSlice()->setExtractionWindow(win);
}

void LayerRGTPropPanelOnSlice::sliceChanged(int value) {
	m_rep->layerSlice()->setSlicePosition(value);
}

