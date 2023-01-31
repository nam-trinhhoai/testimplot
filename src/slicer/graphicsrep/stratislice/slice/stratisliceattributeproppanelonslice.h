#ifndef StratiSliceAttributePropPanelOnSlice_H
#define StratiSliceAttributePropPanelOnSlice_H

#include <QWidget>
#include "pickinginfo.h"
class StratiSliceAttributeRepOnSlice;
class QSlider;
class QSpinBox;
class QToolButton;
class QLineEdit;
class PointPickingTask;

class StratiSliceAttributePropPanelOnSlice: public QWidget {
Q_OBJECT
public:
	StratiSliceAttributePropPanelOnSlice(StratiSliceAttributeRepOnSlice *rep,
			QWidget *parent);
	virtual ~StratiSliceAttributePropPanelOnSlice();

	void setSlicePosition(int pos);
private slots:
	void valueChanged();
	void sliceChanged(int val);

	void extractionWindowChanged(unsigned int size);
	void RGTIsoValueChanged(int pos);

	void pick();
	void pointPicked(double worldX, double worldY, Qt::MouseButton button,
			Qt::KeyboardModifiers keys, const QVector<PickingInfo> &info);
private:
	uint getExtactionWindow();
	void updateSpinValue(int value, QSlider *slider, QSpinBox *spin);

	void createlinkedSliderSpin(QWidget *parent, QSlider *slider,
			QSpinBox *spin);
	QWidget* createSlideSpinBox(QString title, QSlider *slider, QSpinBox *spin);
	QWidget* createWindowParameterWidget();

	void updateSliderSpin(int min, int max, QSlider *slider, QSpinBox *spin);

	void unregisterPickingTask();
protected:
	;
	StratiSliceAttributeRepOnSlice *m_rep;

	//Extraction window
	QLineEdit *m_window;

	//RGT Slicing
	QSlider *m_sliceSlider;
	QSpinBox *m_sliceSpin;

	QToolButton *m_pickButton;
	PointPickingTask *m_pickingTask;
};

#endif
