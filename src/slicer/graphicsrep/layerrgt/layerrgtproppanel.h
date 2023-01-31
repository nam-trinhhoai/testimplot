#ifndef LayerRGTPropPanel_H
#define LayerRGTPropPanel_H

#include <QWidget>

#include "lookuptable.h"

class QLineEdit;
class LayerRGTRep;
class PaletteWidget;
class QSlider;
class QSpinBox;
class QGroupBox;
class QCheckBox;
class AbstractView;
class QToolButton;
class PointPickingTask;

class LayerRGTPropPanel: public QWidget {
	Q_OBJECT
public:
	LayerRGTPropPanel(LayerRGTRep *rep, bool for3D,QWidget *parent);
	virtual ~LayerRGTPropPanel();
	void updatePalette();



	void setNameAttribut(QString);

private slots:
	void valueChanged();
	void sliceChanged(int val);
	void showCrossHair(int value);

	void extractionWindowChanged(unsigned int size);
	void RGTIsoValueChanged(int pos);

	void pick();
	void pointPicked(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys);
	void updateLockCheckBox();
	void lockPalette(int state);
	void updateLockRange(const QVector2D &);
	void updateLockLookupTable(const LookupTable& lookupTable);
protected:
	uint getExtactionWindow();
	QVector2D initSliceRange();
	QWidget* createSliceBox();

	void updateSpinValue(int value,QSlider * slider, QSpinBox * spin);
private:
	QLineEdit *m_window;
	LayerRGTRep *m_rep;
	QSlider *m_sliceSlider;
	QSpinBox *m_sliceSpin;
	QGroupBox *m_sliderBox;
	PaletteWidget *m_palette;

	QCheckBox * m_showCrossHair;
	QCheckBox * m_lockPalette;

	QToolButton * m_pickButton;
	PointPickingTask * m_pickingTask;

	QString m_nameAttribut;
};

#endif /* QTCUDAIMAGEVIEWER_SRC_APP_RGTHORIZONTALSLICEVIEW_H_ */
