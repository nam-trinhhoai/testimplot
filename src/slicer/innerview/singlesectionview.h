#ifndef SingleSectionView_H
#define SingleSectionView_H

#include <QSpinBox>
#include "abstractsectionview.h"
#include "affinetransformation.h"

class QSlider;


//Specialized graphic view to handle section: all the views need to be synchronized
class SingleSectionView: public AbstractSectionView {
Q_OBJECT
public:
	typedef enum SourceType {
		SpinBox, Slider, ExternalCall
	} SourceType;

	SingleSectionView(bool restictToMonoTypeSplit, ViewType type,
			QString uniqueName);
	virtual ~SingleSectionView();

	void updateSlicePosition(int worldVal, int imageVal) override;
	void sliceChangedFromOther(int val, bool isDelta);

	int sliceValueWorld() const
	{
		return m_sliceSpin->value();
	}
	void setSliceValue(int value)
	{
		m_sliceSpin->setValue(value);
	}

protected slots:
	void sliceChangedFromSpinBox(int val);
	void sliceChangedFromSlider(int val);
	void onSliceChangedRequestFromRep(int val);
protected:
	void showRep(AbstractGraphicRep *rep) override;
	void hideRep(AbstractGraphicRep *rep) override;

	QWidget* createSliceBox();

	void defineSliceMinMax(const QVector2D &imageMinMax, int step);
	void defineSliceVal(int imagePos, SourceType source);

	void cleanupRep(AbstractGraphicRep *rep) override;

private:
	void sliceChanged(int val, SourceType source);
	void updateSlicePositionPrivate(int worldVal, int imageVal, SourceType source);

	QSlider *m_sliceImageSlider;
	QSpinBox *m_sliceSpin;
	int m_sliceValueWorld = 0;
	AffineTransformation m_currentSliceTransformation;

};

#endif
