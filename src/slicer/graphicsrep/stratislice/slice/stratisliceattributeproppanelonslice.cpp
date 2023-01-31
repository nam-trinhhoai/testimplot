#include "stratisliceattributeproppanelonslice.h"

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
#include <QMessageBox>
#include <QAction>
#include <QToolButton>

#include <iostream>
#include <sstream>

#include "rgbpalettewidget.h"
#include "cudaimagepaletteholder.h"
#include "cudargbimage.h"
#include "abstractstratisliceattribute.h"
#include "stratisliceattributereponslice.h"
#include "pointpickingtask.h"
#include "abstractinnerview.h"
#include "stratislice.h"
#include "seismic3dabstractdataset.h"

StratiSliceAttributePropPanelOnSlice::StratiSliceAttributePropPanelOnSlice(StratiSliceAttributeRepOnSlice *rep, QWidget *parent) :
		QWidget(parent) {
	m_rep = rep;
	m_pickingTask=nullptr;
	QVBoxLayout *processLayout = new QVBoxLayout(this);
	//processLayout->setMargin(0);
	processLayout->setContentsMargins(0,0,0,0);
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

	m_pickButton = new QToolButton(this);
	QAction *pickAction = new QAction(tr("Pick"), this);
	connect(pickAction, SIGNAL(triggered()), this, SLOT(pick()));
	pickAction->setCheckable(true);
	m_pickButton->setDefaultAction(pickAction);
	processLayout->addWidget(m_pickButton, 0, Qt::AlignmentFlag::AlignTop);

	//Window
	processLayout->addWidget(createWindowParameterWidget(), 0,
			Qt::AlignmentFlag::AlignTop);

	//Colors
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

	//initialize
	extractionWindowChanged(m_rep->stratiSliceAttribute()->extractionWindow());
	RGTIsoValueChanged(m_rep->stratiSliceAttribute()->currentPosition());
}

StratiSliceAttributePropPanelOnSlice::~StratiSliceAttributePropPanelOnSlice() {

}

void StratiSliceAttributePropPanelOnSlice::unregisterPickingTask() {
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

void StratiSliceAttributePropPanelOnSlice::pick() {
	if (m_pickButton->isChecked()) {
		//Register one task only
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
	} else
		unregisterPickingTask();

}

void StratiSliceAttributePropPanelOnSlice::pointPicked(double worldX, double worldY,
		Qt::MouseButton button, Qt::KeyboardModifiers keys,
		const QVector<PickingInfo> &info) {
	if (button != Qt::MouseButton::LeftButton)
		return;
	bool found = false;
	double rgtValue;
	for (PickingInfo i : info) {
		if (i.uuid() == m_rep->stratiSliceAttribute()->stratiSlice()->rgt()->dataID() && !i.value().empty()) {
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
	m_rep->stratiSliceAttribute()->setSlicePosition((int) rgtValue);
}

void StratiSliceAttributePropPanelOnSlice::extractionWindowChanged(unsigned int size) {
	QSignalBlocker b(m_window);
	m_window->setText(
			locale().toString(m_rep->stratiSliceAttribute()->extractionWindow()));
}
void StratiSliceAttributePropPanelOnSlice::RGTIsoValueChanged(int pos) {
	updateSpinValue(pos, m_sliceSlider, m_sliceSpin);
}
QWidget* StratiSliceAttributePropPanelOnSlice::createSlideSpinBox(QString title, QSlider *slider,
		QSpinBox *spin) {
	QGroupBox *sliderBox = new QGroupBox(title, this);
	sliderBox->setContentsMargins(0, 0, 0, 0);

	createlinkedSliderSpin(sliderBox, slider, spin);

	return sliderBox;
}

QWidget* StratiSliceAttributePropPanelOnSlice::createWindowParameterWidget() {
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

void StratiSliceAttributePropPanelOnSlice::updateSpinValue(int value, QSlider *slider,
		QSpinBox *spin) {
	QSignalBlocker b1(slider);
	QSignalBlocker b2(spin);
	slider->setValue(value);
	spin->setValue(value);
}

void StratiSliceAttributePropPanelOnSlice::createlinkedSliderSpin(QWidget *parent,
		QSlider *slider, QSpinBox *spin) {
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

uint StratiSliceAttributePropPanelOnSlice::getExtactionWindow() {
	bool ok;
	uint win = locale().toUInt(m_window->text(), &ok);
	if (!ok)
		return m_rep->stratiSliceAttribute()->extractionWindow();
	return win;
}

void StratiSliceAttributePropPanelOnSlice::sliceChanged(int value) {
	m_rep->stratiSliceAttribute()->setSlicePosition(value);
}

void StratiSliceAttributePropPanelOnSlice::setSlicePosition(int pos) {
	QSignalBlocker b1(m_sliceSlider);
	int realPos = pos;
	if (pos < m_sliceSlider->minimum())
		realPos = m_sliceSlider->minimum();
	if (pos > m_sliceSlider->maximum())
		realPos = m_sliceSlider->maximum();
	m_sliceSlider->setValue(realPos);
}

void StratiSliceAttributePropPanelOnSlice::valueChanged() {
	uint win = getExtactionWindow();
	m_rep->stratiSliceAttribute()->setExtractionWindow(win);
}

void StratiSliceAttributePropPanelOnSlice::updateSliderSpin(int min, int max, QSlider *slider,
		QSpinBox *spin) {
	slider->setMaximum(max);
	slider->setMinimum(min);
	slider->setValue(min);

	spin->setMaximum(max);
	spin->setMinimum(min);
	spin->setValue(min);
}

