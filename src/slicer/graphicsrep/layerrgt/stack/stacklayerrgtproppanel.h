#ifndef StackLayerRGTPropPanel_H
#define StackLayerRGTPropPanel_H

#include <QWidget>

#include "lookuptable.h"

class QLineEdit;
class StackLayerRGTRep;
class PaletteWidget;
class QSlider;
class QSpinBox;
class QGroupBox;
class QCheckBox;
class AbstractView;
class QToolButton;
class PointPickingTask;

class StackLayerRGTPropPanel: public QWidget {
	Q_OBJECT
public:
	StackLayerRGTPropPanel(StackLayerRGTRep *rep, bool for3D,QWidget *parent);
	virtual ~StackLayerRGTPropPanel();
	void updatePalette();
private slots:
	void showCrossHair(int value);
	void updateLockCheckBox();
	void lockPalette(int state);
	void updateLockRange(const QVector2D &);
	void updateLockLookupTable(const LookupTable& lookupTable);


	void pick();
	void pointPicked(double worldX,double worldY,Qt::MouseButton button,Qt::KeyboardModifiers keys);

private:
	QLineEdit *m_window;
	StackLayerRGTRep *m_rep;
	PaletteWidget *m_palette;

	QCheckBox * m_showCrossHair;
	QCheckBox * m_lockPalette;

	QToolButton * m_pickButton;
	PointPickingTask * m_pickingTask;
};

#endif /* QTCUDAIMAGEVIEWER_SRC_APP_RGTHORIZONTALSLICEVIEW_H_ */
