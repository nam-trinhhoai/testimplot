#ifndef LayerRGTPropPanelOnSlice_H
#define LayerRGTPropPanelOnSlice_H

#include <QWidget>
#include "pickinginfo.h"

class QLineEdit;
class LayerRGTRepOnSlice;
class PaletteWidget;
class QSlider;
class QSpinBox;
class QCheckBox;
class AbstractView;
class QToolButton;
class PointPickingTask;

class LayerRGTPropPanelOnSlice: public QWidget {
	Q_OBJECT
public:
	LayerRGTPropPanelOnSlice(LayerRGTRepOnSlice *rep, QWidget *parent);
	virtual ~LayerRGTPropPanelOnSlice();

private slots:
	void valueChanged();
	void sliceChanged(int val);

	void extractionWindowChanged(unsigned int size);
	void RGTIsoValueChanged(int pos);

	void pick();
	void pointPicked(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys,const QVector<PickingInfo> & info);

protected:
	uint getExtactionWindow();
	QVector2D initSliceRange();
	QWidget* createSliceBox();

	void updateSpinValue(int value,QSlider * slider, QSpinBox * spin);

	void unregisterPickingTask();
private:
	QLineEdit *m_window;
	LayerRGTRepOnSlice *m_rep;

	QSlider *m_sliceSlider;
	QSpinBox *m_sliceSpin;

	QToolButton * m_pickButton;
	PointPickingTask * m_pickingTask;
};

#endif /* QTCUDAIMAGEVIEWER_SRC_APP_RGTHORIZONTALSLICEVIEW_H_ */
