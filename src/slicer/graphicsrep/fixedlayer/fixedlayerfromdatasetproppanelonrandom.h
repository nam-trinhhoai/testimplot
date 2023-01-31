#ifndef FixedLayerFromDatasetPropPanelOnRandom_H
#define FixedLayerFromDatasetPropPanelOnRandom_H

#include <QWidget>
#include "pickinginfo.h"

class FixedLayerFromDatasetRepOnRandom;
class QCheckBox;
class AbstractView;
class QPushButton;

class FixedLayerFromDatasetPropPanelOnRandom: public QWidget {
	Q_OBJECT
public:
	FixedLayerFromDatasetPropPanelOnRandom(FixedLayerFromDatasetRepOnRandom *rep, QWidget *parent);
	virtual ~FixedLayerFromDatasetPropPanelOnRandom();

	void setColor(QColor);
private:
	void selectColor();

	FixedLayerFromDatasetRepOnRandom *m_rep;

	QPushButton* m_colorButton;
};

#endif /* QTCUDAIMAGEVIEWER_SRC_APP_RGTHORIZONTALSLICEVIEW_H_ */
