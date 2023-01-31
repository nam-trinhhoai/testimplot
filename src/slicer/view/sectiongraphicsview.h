#ifndef SectionGraphicsView_H
#define SectionGraphicsView_H

#include "viewutils.h"
#include "mousesynchronizedgraphicsview.h"
#include "affinetransformation.h"

class QSlider;
class QSpinBox;
class SectionGraphicsView: public MouseSynchronizedGraphicsView {
Q_OBJECT
public:
	SectionGraphicsView(ViewType type,WorkingSetManager *factory,QString uniqueName,QWidget *parent);
	virtual ~SectionGraphicsView();

protected slots:
	void sliceChanged(int );
	void onSliceChangedRequestFromRep(int );


protected:
	void updateSlicePosition(int worldVal, int imageVal);

	QWidget* createSliceBox(const QString &title);

	void defineSliceMinMax(const QVector2D &imageMinMax, int step);
	void defineSliceVal(int imagePos);

	virtual void showRep(AbstractGraphicRep *rep) override;
	virtual void hideRep(AbstractGraphicRep *rep) override;

private:
	QSlider *m_sliceImageSlider;
	QSpinBox *m_sliceSpin;

	AffineTransformation m_currentSliceTransformation;
	SampleUnit m_sectionType = SampleUnit::NONE;
};


#endif
