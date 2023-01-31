#include "singlesectionview.h"

#include <iostream>

#include <QSpinBox>
#include <QSlider>
#include <QLabel>
#include <QHBoxLayout>


#include "slicerep.h"
#include "isliceablerep.h"
#include "isampledependantrep.h"
#include "cudaimagepaletteholder.h"
#include "GraphicSceneEditor.h"

SingleSectionView::SingleSectionView(bool restictToMonoTypeSplit, ViewType type,
		QString uniqueName) :
		AbstractSectionView(restictToMonoTypeSplit, type, uniqueName) {

	QWidget *sliceBox = createSliceBox();
	sliceBox->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
	m_mainLayout->insertWidget(0,sliceBox);

	connect (m_sliceSpin, SIGNAL(valueChanged(int)), m_scene, SLOT (updateSlice(int)));

}

void SingleSectionView::onSliceChangedRequestFromRep(int val) {
	if (val == m_sliceImageSlider->value())
		return;
	double imageVal;
	m_currentSliceTransformation.indirect(val, imageVal);
	updateSlicePosition(val, (int) imageVal);
}

void SingleSectionView::sliceChangedFromSpinBox(int val) {
	sliceChanged(val, SourceType::SpinBox);
}

void SingleSectionView::sliceChangedFromSlider(int val) {
	sliceChanged(val, SourceType::Slider);
}

void SingleSectionView::sliceChanged(int val, SourceType source) {
	int realVal = val;
	int reste = val % (int) m_currentSliceTransformation.a();
	if (reste != 0) {
		realVal = val + reste;
	}
	double imageVal;
	m_currentSliceTransformation.indirect(realVal, imageVal);

	updateSlicePositionPrivate(realVal, (int) imageVal, source);
	//GS Synchro
//	std::cout << "SINGLE SECTION SLICE CHANGED " << val << "\n";
	emit sliceChangedFromView(realVal, (realVal-m_sliceValueWorld), this);
	m_sliceValueWorld = realVal;
}

void SingleSectionView::sliceChangedFromOther(int val, bool isDelta) {
	int currentVal = m_sliceImageSlider->value();
	if ( isDelta ){
		val+=currentVal;
	}

	int realVal = val;
	int reste = val % (int) m_currentSliceTransformation.a();
	if (reste != 0) {
		realVal = val + reste;
	}
	double imageVal;
	m_currentSliceTransformation.indirect(realVal, imageVal);

	updateSlicePosition(realVal, (int) imageVal);
}

void SingleSectionView::updateSlicePosition(int worldVal, int imageVal) {
	updateSlicePositionPrivate(worldVal, imageVal, SourceType::ExternalCall);
}

void SingleSectionView::updateSlicePositionPrivate(int worldVal, int imageVal, SourceType source) {
	defineSliceVal(worldVal, source);
	AbstractSectionView::updateSlicePosition(worldVal,imageVal);
}

void SingleSectionView::showRep(AbstractGraphicRep *rep) {
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
		slice->setSliceWorldPosition(m_sliceImageSlider->value(),true);

		if (firstSlice() == nullptr) {
			addAxis(slice->image());
			defineScale(slice);
			//updateTile(slice->name());

			m_currentSliceIJPosition=slice->currentSliceIJPosition();
			m_currentSliceWorldPosition=slice->currentSliceWorldPosition();

		}

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
		AbstractSectionView::showRep(rep);
		updateVerticalAxisColor();
		updateTitleFromSlices();
	} else{
		// fail to add
		//qDebug() << "SingleSectionView : fail to add rep " << rep->name() << " error messages : "<< errorMsg;
	}

//	if (SliceRep *slice = dynamic_cast<SliceRep*>(rep)) {
//		for (AbstractGraphicRep *r : m_visibleReps) {
//			if (SliceRep *pSliceRep = dynamic_cast<SliceRep*>(r)) {
//				if(pSliceRep->name().toLower().contains("rgt")){
//					hideRep(r);
//					AbstractSectionView::showRep(r);
//					break;
//				}
//			}
//		}
//	}
}

void SingleSectionView::hideRep(AbstractGraphicRep *rep) {

	if (SliceRep *slice = dynamic_cast<SliceRep*>(rep)) {
		disconnect(slice, SIGNAL(sliceWordPositionChanged(int )), this,
				SLOT(onSliceChangedRequestFromRep(int )));
	}
	AbstractSectionView::hideRep(rep);
	if (firstSlice() == nullptr) {
		removeAxis();
	}
	if (m_visibleReps.count()==0) {
		m_sectionType = SampleUnit::NONE;
	}
	updateTitleFromSlices();
}

void SingleSectionView::cleanupRep(AbstractGraphicRep *rep) {
	AbstractSectionView::cleanupRep(rep);
	if (firstSlice() == nullptr) {
		removeAxis();
	}
	if (m_visibleReps.count()==0) {
		m_sectionType = SampleUnit::NONE;
	}
	updateTitleFromSlices();
}


void SingleSectionView::defineSliceMinMax(const QVector2D &imageMinMax,int step) {
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
	dynamic_cast<GraphicSceneEditor *> (m_scene)->setSliceValue((int) imageMinMax.x());
	m_sliceSpin->setSingleStep(step);
}

void SingleSectionView::defineSliceVal(int image, SourceType source) {
	QSignalBlocker b1(m_sliceImageSlider);
	m_sliceImageSlider->setValue(image);

	if (source!=SourceType::SpinBox) {
		QSignalBlocker b2(m_sliceSpin);
		m_sliceSpin->setValue(image);
	}
}

QWidget* SingleSectionView::createSliceBox() {
	QWidget *controler = new QWidget(this);
	m_sliceImageSlider = new QSlider(Qt::Orientation::Horizontal, this);
	m_sliceImageSlider->setSingleStep(1);
	m_sliceImageSlider->setTracking(true);

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
			SLOT(sliceChangedFromSpinBox(int )));
	connect(m_sliceImageSlider, SIGNAL(valueChanged(int)), this,
			SLOT(sliceChangedFromSlider(int)));

	QHBoxLayout *hBox = new QHBoxLayout(controler);

	hBox->addWidget(m_sliceSpin);
	hBox->addWidget(m_sliceImageSlider);

	QPushButton* paletteButton = getPaletteButton();
	if (paletteButton) {
		hBox->addWidget(paletteButton,0,Qt::AlignRight);
	}
	hBox->addWidget(getSplitButton(),0,Qt::AlignRight);
	return controler;
}

SingleSectionView::~SingleSectionView() {

}
