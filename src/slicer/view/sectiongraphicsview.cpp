#include "sectiongraphicsview.h"

#include <QSlider>
#include <QSpinBox>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QDockWidget>
#include <QLabel>
#include "slavesectionview.h"

#include "slicerep.h"
#include "isliceablerep.h"
#include "isampledependantrep.h"

SectionGraphicsView::SectionGraphicsView(ViewType type,
		WorkingSetManager *factory, QString uniqueName, QWidget *parent) :
		MouseSynchronizedGraphicsView(factory,type, uniqueName, parent) {
	QString title;
	if (type == ViewType::InlineView) {
		title = "Inline";
	} else if (type == ViewType::XLineView) {
		title = "Xline";
	}
	setWindowTitle(title);

	KDDockWidgets::DockWidget *sliceControler = new KDDockWidgets::DockWidget(uniqueName+"_sliceControler");
//	sliceControler->setFeatures(
//			QDockWidget::DockWidgetFeature::DockWidgetFloatable
//					| QDockWidget::DockWidgetFeature::DockWidgetMovable);
	sliceControler->setOptions(KDDockWidgets::DockWidget::Option_NotClosable);
	QWidget *sliceBox = createSliceBox(title);
	sliceBox->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
	sliceControler->setWidget(sliceBox);
	sliceControler->setWindowTitle(title+QString(" controler"));

	addDockWidget(sliceControler, KDDockWidgets::Location_OnRight);

	addDockWidget(m_parametersControler, KDDockWidgets::Location_OnRight, sliceControler);

	AbstractInnerView *firstView = generateView(type,true);
	registerView(firstView);

	addDockWidget(firstView, KDDockWidgets::Location_OnBottom, sliceControler);

	m_sectionType = SampleUnit::NONE;
}

void SectionGraphicsView::showRep(AbstractGraphicRep *rep) {
	bool isAddedCorrectly = true;
	QStringList errorMsg;
	if (SliceRep *slice = dynamic_cast<SliceRep*>(rep)) {
		int currentMin = m_sliceImageSlider->minimum();
		int currentMax = m_sliceImageSlider->maximum();

		QPair<QVector2D, AffineTransformation> sliceRangeAndTransfo =
				slice->sliceRangeAndTransfo();
		QVector2D sliceRange = sliceRangeAndTransfo.first;
		m_currentSliceTransformation = sliceRangeAndTransfo.second;

		if (currentMin != sliceRange.x() || currentMax != sliceRange.y())
			defineSliceMinMax(sliceRange,
					(int) sliceRangeAndTransfo.second.a());
		slice->setSliceWorldPosition(m_sliceImageSlider->value(), true);

		connect(slice, SIGNAL(sliceWordPositionChanged(int )), this,
				SLOT(onSliceChangedRequestFromRep(int )));
	}

	if (ISliceableRep *slice = dynamic_cast<ISliceableRep*>(rep)) {
		double imageVal;
		m_currentSliceTransformation.indirect(m_sliceImageSlider->value(),
				imageVal);
		slice->setSliceIJPosition((int) imageVal);
	}


	ISampleDependantRep* sampleRep = dynamic_cast<ISampleDependantRep*>(rep);
	if (isAddedCorrectly && sampleRep!=nullptr) {
		QList<SampleUnit> units = sampleRep->getAvailableSampleUnits();
		if (m_sectionType==SampleUnit::NONE && units.count()>0) {
			m_sectionType = units[0];
			isAddedCorrectly = sampleRep->setSampleUnit(m_sectionType);
		} else if (m_sectionType!=SampleUnit::NONE && units.contains(m_sectionType)) {
			isAddedCorrectly = sampleRep->setSampleUnit(m_sectionType);
		} else{
			isAddedCorrectly = false;
		}
		if (!isAddedCorrectly && m_sectionType!=SampleUnit::NONE) {
			errorMsg << sampleRep->getSampleUnitErrorMessage(m_sectionType);
		} else if (!isAddedCorrectly) {
			errorMsg << "Display unit unknown";
		}
	}

	if (isAddedCorrectly) {
		MouseSynchronizedGraphicsView::showRep(rep);
	} else{
		// fail to add
		//qDebug() << "SectionGraphicsView : fail to add rep " << rep->name() << " error messages : "<< errorMsg;
	}
}

void SectionGraphicsView::hideRep(AbstractGraphicRep *rep) {
	if (SliceRep *slice = dynamic_cast<SliceRep*>(rep)) {
		disconnect(slice, SIGNAL(sliceWordPositionChanged(int )), this,
				SLOT(onSliceChangedRequestFromRep(int )));
	}

	MouseSynchronizedGraphicsView::hideRep(rep);

	if (innerViews().count()==0) {
		m_sectionType = SampleUnit::NONE;
	}
}

QWidget* SectionGraphicsView::createSliceBox(const QString &title) {
	//m_sliderBox = new QGroupBox(title);

	QWidget * controler=new QWidget(this);
	m_sliceImageSlider = new QSlider(Qt::Orientation::Horizontal, this);
	m_sliceImageSlider->setSingleStep(1);
	m_sliceImageSlider->setTracking(false);

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

	QHBoxLayout *hBox = new QHBoxLayout(controler);
	hBox->addWidget(new QLabel(title));

	hBox->addWidget(m_sliceSpin);
	hBox->addWidget(m_sliceImageSlider);
	return controler;
}

void SectionGraphicsView::defineSliceMinMax(const QVector2D &imageMinMax,
		int step) {
	QSignalBlocker b1(m_sliceImageSlider);
	m_sliceImageSlider->setMinimum((int) imageMinMax.x());
	m_sliceImageSlider->setMaximum((int) imageMinMax.y());
	m_sliceImageSlider->setSingleStep(step);
	int pageStep = (int) ((imageMinMax.y() - imageMinMax.x()) * 5.0 / 100);
	m_sliceImageSlider->setPageStep(pageStep);
	m_sliceImageSlider->setTickInterval(step);

	QSignalBlocker b2(m_sliceSpin);
	m_sliceSpin->setMinimum((int) imageMinMax.x());
	m_sliceSpin->setMaximum((int) imageMinMax.y());
	m_sliceSpin->setSingleStep(step);
}

void SectionGraphicsView::defineSliceVal(int image) {
	QSignalBlocker b1(m_sliceImageSlider);
	m_sliceImageSlider->setValue(image);

	QSignalBlocker b2(m_sliceSpin);
	m_sliceSpin->setValue(image);
}

void SectionGraphicsView::sliceChanged(int val) {
	int realVal = val;
	int reste = val % (int) m_currentSliceTransformation.a();
	if (reste != 0) {
		realVal = val + reste;
	}
	double imageVal;
	m_currentSliceTransformation.indirect(realVal, imageVal);
	updateSlicePosition(realVal, (int) imageVal);
}

void SectionGraphicsView::onSliceChangedRequestFromRep(int val) {
	//The sider control the no re-intrance
	if (val == m_sliceImageSlider->value())
		return;

	double imageVal;
	m_currentSliceTransformation.indirect(val, imageVal);
	updateSlicePosition(val, (int) imageVal);
}

void SectionGraphicsView::updateSlicePosition(int worldVal, int imageVal) {
	defineSliceVal(worldVal);
	QVector<AbstractInnerView*> views = innerViews();
	for (AbstractInnerView *v : views) {
		if (SlaveSectionView *s = dynamic_cast<SlaveSectionView*>(v)) {
			s->updateSlicePosition(worldVal, imageVal);
		}
	}
}

SectionGraphicsView::~SectionGraphicsView() {

}

